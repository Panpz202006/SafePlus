import torch

from safeplus.models.safeplus import SafePlus


def test_detected_likelihood_and_posterior_are_valid():
    model = SafePlus(2, 4)
    hf = torch.tensor([0.2, 0.3, 0.4])
    hd = torch.tensor([0.1, 0.2, 0.5])
    likelihood, posterior = model.likelihood_and_posterior(hf, hd, True, 2)
    assert 0 < likelihood < 1
    assert torch.allclose(posterior.sum(), torch.tensor(1.0), atol=1e-6)


def test_censored_likelihood_includes_no_commission_mass():
    model = SafePlus(2, 4)
    hf = torch.tensor([0.0, 0.0, 0.0])
    hd = torch.tensor([0.5, 0.5, 0.5])
    likelihood, posterior = model.likelihood_and_posterior(hf, hd, False, 2)
    assert torch.allclose(likelihood, torch.tensor(1.0))
    assert torch.allclose(posterior, torch.zeros(3))


def test_detection_cannot_precede_commission():
    model = SafePlus(2, 4)
    hd = torch.tensor([0.5, 0.5, 0.5])
    assert model._detect_at(hd, onset=2, detected_at=1) == 0
