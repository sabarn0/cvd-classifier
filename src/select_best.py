"""
Auto model selection and promotion.

Queries the `cats-vs-dogs-v2` MLflow experiment, finds the best trial by
`val_acc`, then re-trains that configuration for full epochs and saves the
winner as the deployable model artifact.

The saved model.pt is always a CPU state_dict so it can be loaded on edge
devices without CUDA.

Usage:
    # Default: full retrain (10 epochs) of the best HPO config
    python -m src.select_best

    # Quick smoke test
    python -m src.select_best --epochs 2 \\
        --max-train-samples 200 --max-val-samples 50 --max-test-samples 50
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from src.dataset import CatsDogsDataset, IDX_TO_CLASS, get_transforms
from src.inference import save_model_config
from src.model import get_model
from src.train import evaluate_full, plot_confusion_matrix, plot_curves, run_epoch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_best_run(experiment_name: str, tracking_uri: str) -> dict:
    """Query MLflow and return the params of the best run by val_acc."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found. "
            "Did you run `python -m src.tune` first?"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.val_acc > 0",
        order_by=["metrics.val_acc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No completed runs found in experiment. Run tune.py first.")

    best = runs[0]
    logger.info(f"Best run: {best.info.run_id}  val_acc={best.data.metrics.get('val_acc', 'N/A'):.4f}")
    logger.info(f"  params: {best.data.params}")
    return best.data.params


def build_loader(data_dir, split, transform, batch_size, max_samples=None):
    ds = CatsDogsDataset(data_dir, split, transform=transform)
    if max_samples and max_samples < len(ds):
        ds = Subset(ds, list(range(max_samples)))
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=0, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description="Select best model from HPO and retrain")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=7,
                        help="Full-retrain epochs for the winner (default: 7, recommended range: 5-15)")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--model-out", default="models/model.pt")
    parser.add_argument("--config-out", default="models/model_config.json")
    parser.add_argument("--mlflow-experiment", default="cvd-classifier")
    parser.add_argument("--mlflow-tracking-uri", default="mlruns")
    parser.add_argument("--unfreeze-after", type=int, default=4,
                        help="For MobileNetV3Small: unfreeze backbone after N epochs (default: 4)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # --- 1. Find best HPO configuration ---
    best_params = get_best_run(args.mlflow_experiment, args.mlflow_tracking_uri)

    model_name = best_params.get("model_name", "SimpleCNN")
    lr = float(best_params.get("lr", 1e-3))
    batch_size = int(best_params.get("batch_size", 32))
    dropout = float(best_params.get("dropout", 0.2)) if model_name == "MobileNetV3Small" else 0.4

    logger.info(f"Retraining winner: model={model_name} lr={lr:.2e} batch={batch_size}")

    # --- 2. Build data loaders ---
    tr_tfm, ev_tfm = get_transforms(model_name)
    train_loader = build_loader(args.data_dir, "train", tr_tfm, batch_size, args.max_train_samples)
    val_loader = build_loader(args.data_dir, "val", ev_tfm, batch_size, args.max_val_samples)
    test_loader = build_loader(args.data_dir, "test", ev_tfm, batch_size, args.max_test_samples)

    # --- 3. Build model ---
    model_kwargs = {}
    if model_name == "MobileNetV3Small":
        model_kwargs = {"dropout": dropout, "freeze_backbone": True}

    model = get_model(model_name, num_classes=len(IDX_TO_CLASS), **model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Full retrain with MLflow tracking ---
    with mlflow.start_run(run_name=f"champion-{model_name}") as run:
        mlflow.set_tag("promoted", "true")
        mlflow.set_tag("source", "select_best")
        mlflow.log_params({
            "model_name": model_name,
            "lr": lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "epochs": args.epochs,
            "device": device,
            "img_size": 224,
        })

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, args.epochs + 1):

            # Unfreeze MobileNetV3 backbone halfway through for full fine-tuning
            if model_name == "MobileNetV3Small" and epoch == args.unfreeze_after + 1:
                model.unfreeze_backbone()
                # Re-init optimizer to include newly unfrozen params
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(trainable, lr=lr * 0.1)  # lower LR for backbone
                logger.info(f"Epoch {epoch}: Unfroze MobileNetV3Small backbone (lr={lr*0.1:.2e})")
            elif epoch == 1:
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(trainable, lr=lr)

            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch}/{args.epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # Track best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best checkpoint
        model.load_state_dict(best_state)
        model.cpu()

        # --- 5. Test-set evaluation ---
        test_device = "cpu"
        model.to(test_device)
        y_true, y_pred = evaluate_full(model, test_loader, test_device)
        test_acc = sum(int(p == t) for p, t in zip(y_pred, y_true)) / max(len(y_true), 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        mlflow.log_metrics({
            "test_acc": test_acc, "test_precision": precision,
            "test_recall": recall, "test_f1": f1,
        })
        logger.info(f"Test: acc={test_acc:.4f} p={precision:.4f} r={recall:.4f} f1={f1:.4f}")

        # --- 6. Artifacts ---
        Path("artifacts").mkdir(exist_ok=True)
        class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_path = "artifacts/confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, cm_path)
        mlflow.log_artifact(cm_path)

        curves_path = "artifacts/loss_accuracy_curves.png"
        plot_curves(history, curves_path)
        mlflow.log_artifact(curves_path)

        # Log model to MLflow
        mlflow.pytorch.log_model(model, artifact_path="model")

        # --- 7. Save CPU state_dict for edge deployment ---
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_out)

        # Save config including model_name so inference.py loads the right class
        save_model_config(args.config_out, extra={
            "model_name": model_name,
            "test_acc": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "best_val_acc": best_val_acc,
            "trained_with": {
                "lr": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "epochs": args.epochs,
            },
        })
        mlflow.log_artifact(args.model_out)
        mlflow.log_artifact(args.config_out)

        logger.info(f"Champion model saved: {args.model_out}")
        logger.info(f"Champion config saved: {args.config_out}")
        logger.info(f"MLflow run: {run.info.run_id} (tagged promoted=true)")


if __name__ == "__main__":
    main()
