import json

import numpy as np
import pytest
import torch
import yaml

from safeplus.cli import main


@pytest.mark.parametrize("length_dtype", [np.int64, np.int16, np.uint32])
def test_cli_workflow(tmp_path, capsys, length_dtype):
    data = tmp_path / "data.npz"
    main(["synthetic", "--output", str(data), "--n", "40", "--length", "6", "--features", "2"])
    config = {
        "seed": 0,
        "data": {"name": "test", "path": str(data), "split": [0.6, 0.2, 0.2]},
        "model": {"name": "safeplus", "hidden_dim": 4},
        "training": {
            "batch_size": 16,
            "epochs": 5,
            "patience": 2,
            "device": "cpu",
            "learning_rate": 0.001,
        },
        "output_dir": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        main(["train", "--config", str(path), "--epochs", "1"])
        run = tmp_path / "runs" / "seed_0"
        assert (run / "model.pt").exists()
        assert yaml.safe_load((run / "config.yaml").read_text())["training"]["epochs"] == 1
        assert len(json.loads((run / "history.json").read_text())) == 1
        capsys.readouterr()
        main(["evaluate", "--run-dir", str(run)])
        assert "auprc" in json.loads(capsys.readouterr().out)
        with np.load(data) as content:
            unlabeled = tmp_path / "unlabeled.npz"
            lengths = content["lengths"].astype(length_dtype)
            lengths[0] = 3
            np.savez(unlabeled, x=content["x"], lengths=lengths)
        output = tmp_path / "predictions.npz"
        main(
            [
                "predict",
                "--run-dir",
                str(run),
                "--data",
                str(unlabeled),
                "--output",
                str(output),
                "--batch-size",
                "7",
            ]
        )
        with np.load(output) as content:
            assert content["risk"].shape == (40, 6)
            assert np.isfinite(content["risk"]).all()
            assert np.all(content["risk"][0, 2:] == content["risk"][0, 2])
            np.testing.assert_array_equal(
                content["prediction"], content["final_risk"] >= content["threshold"]
            )
        main(["aggregate", "--runs", str(tmp_path / "runs"), "--output", str(tmp_path / "reports")])
        assert (tmp_path / "reports" / "summary.csv").exists()
        with pytest.raises(SystemExit) as error:
            main(["train", "--config", str(path)])
        assert error.value.code == 2
    finally:
        torch.set_num_threads(threads)
