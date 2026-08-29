import io

import torch
from PIL import Image

from src.inference import preprocess_image


def _make_dummy_image_bytes(size=(300, 150), color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_output_shape():
    image_bytes = _make_dummy_image_bytes()
    tensor = preprocess_image(image_bytes)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_image_dtype_and_normalization_range():
    image_bytes = _make_dummy_image_bytes()
    tensor = preprocess_image(image_bytes)

    assert tensor.dtype == torch.float32
    # With mean=0.5, std=0.5 normalization, pixel values should fall in [-1, 1]
    assert tensor.min() >= -1.0001
    assert tensor.max() <= 1.0001


def test_preprocess_image_handles_non_square_input():
    image_bytes = _make_dummy_image_bytes(size=(500, 100))
    tensor = preprocess_image(image_bytes)
    assert tensor.shape[-2:] == (224, 224)
