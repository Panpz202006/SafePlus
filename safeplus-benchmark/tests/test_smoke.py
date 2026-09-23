import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from safeplus.data.dataset import SequenceDataset
from safeplus.data.synthetic import generate_synthetic
from safeplus.models import SafePlus


def test_one_training_step():
    path = os.getenv("SAFEPLUS_SMOKE_DATA")
    if path:
        with np.load(path) as content:
            arrays = {key: content[key] for key in content.files}
    else:
        arrays = generate_synthetic(n=24, length=8, features=3, seed=4)
    batch = next(iter(DataLoader(SequenceDataset(arrays), batch_size=8)))
    model = SafePlus(input_dim=arrays["x"].shape[-1], hidden_dim=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, output = model.loss(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert output["onset_posterior"].shape == (8, arrays["x"].shape[1])
