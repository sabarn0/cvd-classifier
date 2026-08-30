"""
Optuna hyperparameter optimisation for Cats vs Dogs classifiers.

Searches over:
  - model_name  : SimpleCNN | MobileNetV3Small
  - lr          : log-uniform in [1e-4, 1e-2]
  - batch_size  : 16 | 32 | 64
  - dropout     : uniform in [0.1, 0.5]  (only for MobileNetV3Small)

Each trial trains for `--epochs-per-trial` epochs (default: 3) on GPU.
Each trial is logged as its own MLflow run with full per-epoch metrics:
  train_loss, val_loss, train_acc, val_acc — all visible in the MLflow UI.

Usage:
    # Full run (~50 min on RTX 3060, 20 trials, 3 epochs each)
    python -m src.tune

    # Quick smoke test
    python -m src.tune --n-trials 3 --epochs-per-trial 1 \\
        --max-train-samples 200 --max-val-samples 50

After the study completes, run src/select_best.py to retrain and promote the winner.
"""
import argparse
import logging

import mlflow
import mlflow.pytorch
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from src.dataset import CatsDogsDataset, IDX_TO_CLASS, get_transforms
from src.model import get_model
from src.train import evaluate_full, run_epoch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_loaders(data_dir, model_name, batch_size, max_train, max_val, max_test=None):
    tr_tfm, ev_tfm = get_transforms(model_name)

    train_ds = CatsDogsDataset(data_dir, "train", transform=tr_tfm)
    val_ds = CatsDogsDataset(data_dir, "val", transform=ev_tfm)
    test_ds = CatsDogsDataset(data_dir, "test", transform=ev_tfm)

    if max_train and max_train < len(train_ds):
        train_ds = Subset(train_ds, list(range(max_train)))
    if max_val and max_val < len(val_ds):
        val_ds = Subset(val_ds, list(range(max_val)))
    if max_test and max_test < len(test_ds):
        test_ds = Subset(test_ds, list(range(max_test)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def make_objective(args, device):
    def objective(trial: optuna.Trial) -> float:
        # --- Sample hyperparameters ---
        model_name = trial.suggest_categorical("model_name", ["SimpleCNN", "MobileNetV3Small"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model_kwargs = {}
        if model_name == "MobileNetV3Small":
            model_kwargs["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
            model_kwargs["freeze_backbone"] = True  # keep frozen during HPO for speed

        train_loader, val_loader, test_loader = build_loaders(
            args.data_dir, model_name, batch_size,
            args.max_train_samples, args.max_val_samples, args.max_test_samples,
        )

        model = get_model(model_name, num_classes=len(IDX_TO_CLASS), **model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

        best_val_acc = 0.0

        # Each trial gets its own MLflow run — full per-epoch metrics logged
        # Run name: model-1-ep0, model-2-ep0, ...
        run_name = f"model-{trial.number + 1}-ep0"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "trial_number": trial.number + 1,
                "model_name": model_name,
                "lr": lr,
                "batch_size": batch_size,
                "dropout": model_kwargs.get("dropout", "N/A"),
                "epochs_per_trial": args.epochs_per_trial,
                "device": device,
            })

            for epoch in range(1, args.epochs_per_trial + 1):
                train_loss, train_acc = run_epoch(
                    model, train_loader, criterion, optimizer, device, train=True)
                val_loss, val_acc = run_epoch(
                    model, val_loader, criterion, optimizer, device, train=False)

                # Log all 4 metrics per epoch — all visible in MLflow UI
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "train_acc":  train_acc,
                    "val_acc":    val_acc,
                }, step=epoch)

                trial.report(val_acc, epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    raise optuna.exceptions.TrialPruned()

                best_val_acc = max(best_val_acc, val_acc)
                logger.info(
                    f"Trial {trial.number} Epoch {epoch}/{args.epochs_per_trial} "
                    f"model={model_name} lr={lr:.2e} bs={batch_size} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f}   val_acc={val_acc:.4f}"
                )

            # Test-set evaluation for this trial
            y_true, y_pred = evaluate_full(model, test_loader, device)
            test_acc = sum(int(p == t) for p, t in zip(y_pred, y_true)) / max(len(y_true), 1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            mlflow.log_metrics({
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            })

        return best_val_acc

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for Cats vs Dogs")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20, ~50 min on RTX 3060)")
    parser.add_argument("--timeout", type=int, default=3000,
                        help="Hard timeout in seconds (default: 3000 = ~50 min)")
    parser.add_argument("--epochs-per-trial", type=int, default=3,
                        help="Epochs per trial (default: 3)")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--mlflow-experiment", default="cvd-classifier")
    parser.add_argument("--mlflow-tracking-uri", default="mlruns")
    parser.add_argument("--study-name", default="cvd-hpo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # Seeded TPE sampler — reproducible trial ordering across runs.
    # n_startup_trials=5: random exploration for first 5 trials, then
    # Bayesian exploitation kicks in (important with only 20 trials total).
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5)

    # Median pruner: prune trials performing below median at each intermediate step
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=None,  # in-memory; all results persisted via MLflow
    )

    study.optimize(
        make_objective(args, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info("=" * 60)
    logger.info(f"Best trial : #{best.number}")
    logger.info(f"  val_acc  : {best.value:.4f}")
    logger.info(f"  params   : {best.params}")
    logger.info("=" * 60)
    logger.info("Run `python -m src.select_best` to retrain and promote the winner.")


if __name__ == "__main__":
    main()
