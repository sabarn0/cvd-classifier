"""
Model-loading and prediction utilities shared by the FastAPI service and the
unit tests. Keeping this logic separate from api/main.py makes it easy to
unit-test without spinning up FastAPI.
"""
import io
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import IDX_TO_CLASS, eval_transform
from src.model import SimpleCNN


def load_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a SimpleCNN with weights from `model_path` (a state_dict .pt file)."""
    model = SimpleCNN(num_classes=len(IDX_TO_CLASS))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Data pre-processing function: raw image bytes -> normalized tensor
    of shape (1, 3, 224, 224), ready for model input.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = eval_transform(image)
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
