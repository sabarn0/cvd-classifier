import torch

from src.dataset import IDX_TO_CLASS
from src.inference import predict
from src.model import SimpleCNN


def test_predict_returns_valid_label_and_probabilities():
    torch.manual_seed(0)
    model = SimpleCNN(num_classes=len(IDX_TO_CLASS))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    label, probabilities = predict(model, dummy_input)

    assert label in IDX_TO_CLASS.values()
    assert set(probabilities.keys()) == set(IDX_TO_CLASS.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-3
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())


def test_predict_label_matches_argmax_probability():
    torch.manual_seed(1)
    model = SimpleCNN(num_classes=len(IDX_TO_CLASS))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    label, probabilities = predict(model, dummy_input)

    assert label == max(probabilities, key=probabilities.get)


def test_predict_on_batch_of_one_is_deterministic_in_eval_mode():
    torch.manual_seed(2)
    model = SimpleCNN(num_classes=len(IDX_TO_CLASS))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    label1, probs1 = predict(model, dummy_input)
    label2, probs2 = predict(model, dummy_input)

    assert label1 == label2
    assert probs1 == probs2
