import glob
import os
import random
import pytest
import requests

DATA_DIR = "data/processed"
DEFAULT_HOST = os.environ.get("TEST_HOST", "http://localhost:7000")


@pytest.fixture(scope="module")
def random_test_images():
    """Sample 25 random images from the test split."""
    cat_images = glob.glob(os.path.join(DATA_DIR, "test", "cat", "*.jpg"))
    dog_images = glob.glob(os.path.join(DATA_DIR, "test", "dog", "*.jpg"))
    if not cat_images or not dog_images:
        pytest.skip("Processed dataset not found at data/processed/test")

    # Sample 12 cats and 13 dogs for 25 total
    samples = [(p, "cat") for p in random.sample(cat_images, 12)] + \
              [(p, "dog") for p in random.sample(dog_images, 13)]
    random.shuffle(samples)
    return samples


def test_endpoint_predict_batch_25_random_images(random_test_images):
    """Test /predict with 25 random images and assert accuracy is above 80%."""
    correct = 0

    for img_path, true_label in random_test_images:
        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{DEFAULT_HOST}/predict",
                files={"file": (os.path.basename(img_path), f, "image/jpeg")},
                timeout=10
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "label" in data
        assert "probabilities" in data
        assert "latency_ms" in data
        assert data["label"] in {"cat", "dog"}

        if data["label"] == true_label:
            correct += 1

    accuracy = correct / len(random_test_images)
    print(f"\nBatch Endpoint Test: {correct}/25 correct ({accuracy * 100:.1f}% accuracy)")
    # Model has 96% accuracy on test set, so batch accuracy should be high (>= 80%)
    assert accuracy >= 0.80, f"Batch accuracy {accuracy * 100:.1f}% is lower than required 80% threshold."
