"""
Model-loading and prediction utilities shared by the FastAPI service and the
unit tests. Keeping this logic separate from api/main.py makes it easy to
unit-test without spinning up FastAPI.

v2 changes:
  - load_model() reads model_config.json to determine the correct architecture
    (SimpleCNN or MobileNetV3Small), so the API transparently serves either.
  - preprocess_image() accepts an optional model_name to use the right transforms.
"""
import io
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import IDX_TO_CLASS, eval_transform, imagenet_eval_transform, get_transforms
from src.model import get_model, SimpleCNN


def load_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load a model with weights from `model_path` (a state_dict .pt file).

    The model architecture is determined by reading the sibling model_config.json.
    Falls back to SimpleCNN if no config is found (backwards compatibility with main).
    """
    model_path = Path(model_path)
    config_path = model_path.parent / "model_config.json"

    model_name = "SimpleCNN"
    model_kwargs = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
        model_name = config.get("model_name", "SimpleCNN")
        # Restore dropout if saved (MobileNetV3Small)
        trained_with = config.get("trained_with", {})
        if model_name == "MobileNetV3Small":
            model_kwargs["dropout"] = trained_with.get("dropout", 0.2)
            model_kwargs["freeze_backbone"] = False  # inference: all weights active

    model = get_model(model_name, num_classes=len(IDX_TO_CLASS), **model_kwargs)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _get_model_name_from_path(model_path: str) -> str:
    """Read model_name from sibling model_config.json, default 'SimpleCNN'."""
    config_path = Path(model_path).parent / "model_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return config.get("model_name", "SimpleCNN")
    return "SimpleCNN"


def preprocess_image(image_bytes: bytes, model_name: str = "SimpleCNN") -> torch.Tensor:
    """
    Data pre-processing function: raw image bytes -> normalized tensor
    of shape (1, 3, 224, 224), ready for model input.

    Uses ImageNet normalization for MobileNetV3Small, [-1,1] for SimpleCNN.
    """
    _, ev_tfm = get_transforms(model_name)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = ev_tfm(image)
    return tensor.unsqueeze(0)


def predict(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[str, dict]:
    """
    Model utility / inference function: runs a forward pass and returns
    (predicted_label, {class_name: probability, ...}).
    """
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    label = IDX_TO_CLASS[pred_idx]
    prob_dict = {IDX_TO_CLASS[i]: round(float(probs[i]), 4) for i in range(len(probs))}
    return label, prob_dict


def save_model_config(config_path: str, extra: dict = None):
    config = {"classes": IDX_TO_CLASS, "img_size": 224}
    if extra:
        config.update(extra)
    Path(config_path).write_text(json.dumps(config, indent=2))
