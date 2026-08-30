"""
Automated endpoint validation: samples random images from the test/val split,
sends them to /predict, and validates the returned label against ground truth.

Usage:
    # Test local endpoint on port 7000 (25 random images)
    python scripts/test_batch_endpoint.py --host http://localhost:7000 --n-samples 25

    # Test GCP deployed VM
    python scripts/test_batch_endpoint.py --host http://34.100.156.243:7000 --n-samples 25
"""
import argparse
import glob
import os
import random
import sys
import requests


def run_validation(host: str, n_samples: int = 25, data_dir: str = "data/processed", split: str = "test"):
    cat_images = glob.glob(os.path.join(data_dir, split, "cat", "*.jpg"))
    dog_images = glob.glob(os.path.join(data_dir, split, "dog", "*.jpg"))

    if not cat_images or not dog_images:
        print(f"Error: Could not find images under {data_dir}/{split}/")
        sys.exit(1)

    # Sample balanced dataset
    n_cat = n_samples // 2
    n_dog = n_samples - n_cat
    samples = [(p, "cat") for p in random.sample(cat_images, min(n_cat, len(cat_images)))] + \
              [(p, "dog") for p in random.sample(dog_images, min(n_dog, len(dog_images)))]
    random.shuffle(samples)

    print(f"=== Testing Endpoint: {host}/predict with {len(samples)} random images ===")
    print(f"{'#':<3} | {'True Label':<10} | {'Predicted':<10} | {'Cat Prob':<9} | {'Dog Prob':<9} | {'Latency':<8} | {'Status'}")
    print("-" * 75)

    correct = 0
    total_latency = 0.0

    for idx, (img_path, true_label) in enumerate(samples, 1):
        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{host}/predict",
                files={"file": (os.path.basename(img_path), f, "image/jpeg")},
                timeout=10
            )

        if resp.status_code != 200:
            print(f"{idx:<3} | {true_label:<10} | {'HTTP ' + str(resp.status_code):<10} | {'-':<9} | {'-':<9} | {'-':<8} | ❌ FAILED")
            continue

        data = resp.json()
        pred_label = data.get("label", "unknown")
        probs = data.get("probabilities", {})
        cat_prob = probs.get("cat", 0.0)
        dog_prob = probs.get("dog", 0.0)
        latency = data.get("latency_ms", 0.0)
        total_latency += latency

        is_match = (pred_label == true_label)
        if is_match:
            correct += 1
            status_icon = "MATCH"
        else:
            status_icon = "MISMATCH"

        print(f"{idx:<3} | {true_label:<10} | {pred_label:<10} | {cat_prob:<9.4f} | {dog_prob:<9.4f} | {latency:<6.1f}ms | {status_icon}")

    accuracy = (correct / len(samples)) * 100
    avg_latency = total_latency / len(samples)

    print("-" * 75)
    print(f"Summary: {correct}/{len(samples)} Correct ({accuracy:.1f}% Accuracy)")
    print(f"Average Latency: {avg_latency:.2f} ms")

    if accuracy >= 80.0:
        print("\n>> Endpoint validation PASSED successfully!")
        return 0
    else:
        print(f"\n>> Endpoint validation FAILED: Accuracy {accuracy:.1f}% is below 80% threshold.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test /predict endpoint with random dataset images")
    parser.add_argument("--host", default="http://localhost:7000", help="API base URL (default: http://localhost:7000)")
    parser.add_argument("--n-samples", type=int, default=25, help="Number of test images (default: 25)")
    parser.add_argument("--data-dir", default="data/processed", help="Dataset directory")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"], help="Dataset split to sample from")
    args = parser.parse_args()

    sys.exit(run_validation(args.host, args.n_samples, args.data_dir, args.split))
