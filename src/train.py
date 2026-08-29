"""
Train the baseline CNN on the processed Cats vs Dogs dataset, tracking every
run with MLflow (params, per-epoch metrics, confusion matrix, loss curves,
and the model artifact itself).

Run (after `mlflow ui` is running, or with a MLFLOW_TRACKING_URI set):
    python -m src.train --epochs 5 --batch-size 32

For a quick smoke test without the full dataset:
    python -m src.train --epochs 1 --max-train-samples 200 --max-val-samples 50
"""
import argparse
import json
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

from src.dataset import CatsDogsDataset, IDX_TO_CLASS, eval_transform, train_transform
from src.inference import save_model_config
from src.model import SimpleCNN


def get_loader(data_dir, split, transform, batch_size, shuffle, max_samples=None):
    ds = CatsDogsDataset(data_dir, split, transform=transform)
    if max_samples is not None and max_samples < len(ds):
        ds = Subset(ds, list(range(max_samples)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def evaluate_full(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--model-out", default="models/model.pt")
    parser.add_argument("--config-out", default="models/model_config.json")
    parser.add_argument("--mlflow-experiment", default="cats-vs-dogs")
    parser.add_argument("--mlflow-tracking-uri", default="mlruns")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    train_loader = get_loader(args.data_dir, "train", train_transform, args.batch_size, True, args.max_train_samples)
    val_loader = get_loader(args.data_dir, "val", eval_transform, args.batch_size, False, args.max_val_samples)
    test_loader = get_loader(args.data_dir, "test", eval_transform, args.batch_size, False, args.max_test_samples)

    model = SimpleCNN(num_classes=len(IDX_TO_CLASS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": device,
            "model": "SimpleCNN",
            "img_size": 224,
        })

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }, step=epoch)

            print(f"Epoch {epoch}/{args.epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Final test-set evaluation
        y_true, y_pred = evaluate_full(model, test_loader, device)
        test_acc = sum(int(p == t) for p, t in zip(y_pred, y_true)) / max(len(y_true), 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        mlflow.log_metrics({
            "test_acc": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        })
        print(f"Test: acc={test_acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

        # Artifacts: confusion matrix + loss/accuracy curves
        Path("artifacts").mkdir(exist_ok=True)
        class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_path = "artifacts/confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, cm_path)
        mlflow.log_artifact(cm_path)

        curves_path = "artifacts/loss_accuracy_curves.png"
        plot_curves(history, curves_path)
        mlflow.log_artifact(curves_path)

        # Log the model to MLflow (versioned artifact store)
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Also save a plain state_dict + config for the FastAPI service to load
        # directly without depending on the MLflow artifact store at serve time.
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_out)
        save_model_config(args.config_out, extra={
            "test_acc": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        })
        mlflow.log_artifact(args.model_out)
        mlflow.log_artifact(args.config_out)

        print(f"\nSaved model weights to {args.model_out}")
        print(f"Saved model config to {args.config_out}")
        print(f"MLflow run logged under experiment '{args.mlflow_experiment}' "
              f"(tracking uri: {args.mlflow_tracking_uri})")


if __name__ == "__main__":
    main()
