import pytest
import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.utils.losses import dft_loss, get_supervised_loss_fn, sft_loss


def _padded_batch():
    return {
        "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        "pad_mode": DatasetPadMode.RIGHT,
        "batch_num_tokens": 3,
        "dp_size": 1,
    }


def test_sft_loss_is_mean_target_negative_log_likelihood():
    log_prob = torch.tensor([[-0.2, -0.7, -3.0], [-0.4, -2.0, -1.0]], requires_grad=True)

    loss, metrics = sft_loss(config=None, model_output={"log_probs": log_prob}, data=_padded_batch())

    expected = -torch.tensor([-0.2, -0.7, -0.4]).mean()
    torch.testing.assert_close(loss, expected)
    assert metrics == {}


def test_dft_loss_uses_detached_model_probability_as_token_weight():
    log_prob = torch.tensor([[-0.2, -0.7, -3.0], [-0.4, -2.0, -1.0]], requires_grad=True)

    loss, metrics = dft_loss(config=None, model_output={"log_probs": log_prob}, data=_padded_batch())

    mask = _padded_batch()["response_mask"]
    expected = (-torch.exp(log_prob).detach() * log_prob)[mask].mean()
    torch.testing.assert_close(loss, expected)
    assert metrics == {}

    loss.backward()
    expected_grad = -torch.exp(log_prob.detach()) * mask / mask.sum()
    torch.testing.assert_close(log_prob.grad, expected_grad)


def test_dft_loss_supports_no_padding_nested_tensors():
    log_prob_values = torch.tensor([-0.2, -0.7, -3.0, -0.4, -2.0])
    offsets = torch.tensor([0, 3, 5])
    log_prob = torch.nested.nested_tensor_from_jagged(log_prob_values, offsets)
    loss_mask = torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1, 1, 0, 1], dtype=torch.bool), offsets)
    data = {
        "loss_mask": loss_mask,
        "pad_mode": DatasetPadMode.NO_PADDING,
        "batch_num_tokens": 3,
        "dp_size": 1,
    }

    loss, _ = dft_loss(config=None, model_output={"log_probs": log_prob}, data=data)

    shifted_mask = torch.roll(loss_mask.values(), shifts=-1, dims=0)
    expected = (-torch.exp(log_prob_values) * log_prob_values)[shifted_mask].mean()
    torch.testing.assert_close(loss, expected)


def test_supervised_loss_selector_rejects_unknown_loss():
    assert get_supervised_loss_fn("SFT") is sft_loss
    assert get_supervised_loss_fn("dft") is dft_loss
    with pytest.raises(ValueError, match="trainer.loss_type"):
        get_supervised_loss_fn("unknown")
