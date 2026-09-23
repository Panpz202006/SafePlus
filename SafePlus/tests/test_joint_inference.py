import itertools

import pytest
import torch

from safeplus.models import build_model
from safeplus.models.safeplus import SafePlus


def batch():
    return {
        "x": torch.randn(3, 6, 2),
        "lengths": torch.tensor([6, 4, 2]),
        "detected": torch.tensor([True, False, True]),
        "t_detect": torch.tensor([4, 3, 1]),
    }


def test_enumeration_and_total_probability():
    model = SafePlus(2, 4)
    for values in itertools.product([0.0, 0.3, 1.0], repeat=4):
        hf, hd = torch.tensor(values[:2]), torch.tensor(values[2:])
        mass = model._event_mass(hf)
        events = []
        for end in range(2):
            actual, post = model.likelihood_and_posterior(hf, hd, True, end)
            joint = torch.stack([mass[k] * model._detect_at(hd, k, end) for k in range(end + 1)])
            torch.testing.assert_close(actual, joint.sum())
            if actual > 0:
                torch.testing.assert_close(post, joint / actual)
            events.append(actual)
        censored, post = model.likelihood_and_posterior(hf, hd, False, 1)
        expected = (1 - hf).prod() + sum(
            mass[k] * model._undetected_through(hd, k, 1) for k in range(2)
        )
        torch.testing.assert_close(censored, expected)
        torch.testing.assert_close(sum(events) + censored, torch.tensor(1.0))


def test_batched_inference_and_padding():
    model = SafePlus(2, 4)
    b = batch()
    loss, out = model.loss(b)
    torch.testing.assert_close(out["latent_posterior"].sum(1), torch.ones(3))
    assert out["no_commission_posterior"][0] == 0
    assert out["no_commission_posterior"][1] > 0
    for i in range(3):
        end = int(b["t_detect"][i] if b["detected"][i] else b["lengths"][i] - 1)
        likelihood, posterior = model.likelihood_and_posterior(
            out["commission_hazard"][i], out["detection_hazard"][i], bool(b["detected"][i]), end
        )
        torch.testing.assert_close(out["log_likelihood"][i], likelihood.log())
        torch.testing.assert_close(out["onset_posterior"][i, : end + 1], posterior)
        assert (out["onset_posterior"][i, end + 1 :] == 0).all()
    b["x"][1, 4:] = 999
    torch.testing.assert_close(model.loss(b)[0], loss)
    torch.testing.assert_close(out["risk"][1, 3].expand(3), out["risk"][1, 3:])


@pytest.mark.parametrize("inference", ["exact", "stochastic_em"])
def test_extreme_logits_have_finite_nonzero_gradients(inference):
    model = SafePlus(2, 4, inference=inference, em_samples=4)
    with torch.no_grad():
        model.commission_head.bias.fill_(-100)
        model.detection_head.bias.fill_(-100)
    loss, out = model.loss(batch())
    assert torch.isfinite(loss)
    loss.backward()
    for head in [model.commission_head, model.detection_head]:
        assert torch.isfinite(head.bias.grad).all()
        assert head.bias.grad.abs().sum() > 0
    model.eval()
    loss, out = model.loss(batch())
    torch.testing.assert_close(loss, -out["log_likelihood"].mean())


def test_no_onset_supervision_or_future_leakage():
    model = SafePlus(2, 4)
    b = batch()
    before = model.loss(b)[0]
    b["t_onset"] = torch.tensor([1, 2, 0])
    torch.testing.assert_close(model.loss(b)[0], before)
    risk = model(b["x"], b["lengths"])["risk"]
    b["x"][:, 2:] = 50
    torch.testing.assert_close(model(b["x"], b["lengths"])["risk"][:, :2], risk[:, :2])


def test_safer_matches_event_likelihood():
    model = build_model("safe-r", 2, 4)
    b = batch()
    loss, out = model.loss(b)
    h = out["logits"].sigmoid()
    expected = torch.stack(
        [(1 - h[0, :4]).prod() * h[0, 4], (1 - h[1, :4]).prod(), (1 - h[2, :1]).prod() * h[2, 1]]
    )
    torch.testing.assert_close(loss, -expected.log().mean())
