"""Command-line workflows for the SAFE+ benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from safeplus.data import load_npz, make_loaders
from safeplus.data.synthetic import generate_synthetic
from safeplus.evaluation.metrics import evaluate_predictions, prefix_metrics
from safeplus.models import build_model
from safeplus.training import Trainer
from safeplus.utils import load_config, seed_everything, write_json


def select_threshold(labels, risk):
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [evaluate_predictions(labels, risk, t)["f1"] for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def train(args):
    cfg = load_config(args.config)
    for key in ("epochs", "batch_size", "device"):
        value = getattr(args, key, None)
        if value is not None:
            cfg["training"][key] = value
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "output_dir", None) is not None:
        cfg["output_dir"] = args.output_dir
    if cfg["training"]["epochs"] < 1 or cfg["training"]["batch_size"] < 1:
        raise ValueError("epochs and batch-size must be positive")
    run_dir = Path(cfg["output_dir"]) / f"seed_{cfg['seed']}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Run directory is not empty: {run_dir}; choose --output-dir or --seed")
    seed_everything(cfg["seed"])
    dataset = load_npz(cfg["data"]["path"])
    train_loader, val_loader, test_loader = make_loaders(
        dataset, cfg["training"]["batch_size"], cfg["data"]["split"], cfg["seed"]
    )
    if any(len(loader.dataset) == 0 for loader in (train_loader, val_loader, test_loader)):
        raise ValueError("Each train/validation/test split must contain at least one entity")
    model_options = dict(cfg["model"])
    model_name = model_options.pop("name")
    model = build_model(model_name, dataset.arrays["x"].shape[-1], **model_options)
    trainer = Trainer(model, cfg["training"]["device"], cfg["training"]["learning_rate"])
    history = trainer.fit(
        train_loader, val_loader, cfg["training"]["epochs"], cfg["training"]["patience"]
    )
    val = trainer.run_epoch(val_loader, train=False)
    threshold = select_threshold(val["labels"], val["risk"])
    test = trainer.run_epoch(test_loader, train=False)
    metrics = evaluate_predictions(
        test["labels"],
        test["risk"],
        threshold,
        test["t_detect"],
        test["t_onset"],
        test.get("onset_posterior"),
    )
    metrics.update(prefix_metrics(test["labels"], test["risk"], threshold))
    run_dir = Path(cfg["output_dir"]) / f"seed_{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metrics.json", {"threshold": threshold, **metrics})
    print(run_dir / "metrics.json")


def aggregate(args):
    import pandas as pd

    rows = []
    for metric_path in Path(args.runs).glob("**/metrics.json"):
        config_path = metric_path.parent / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text())
        rows.append(
            {
                "dataset": cfg["data"]["name"],
                "model": cfg["model"]["name"],
                "seed": cfg["seed"],
                **json.loads(metric_path.read_text()),
            }
        )
    if not rows:
        raise ValueError("No metrics.json files found")
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.select_dtypes("number").columns if c != "seed"]
    summary = frame.groupby(["dataset", "model"])[numeric].agg(["mean", "std"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "summary.csv")
    (output / "summary.tex").write_text(summary.to_latex(float_format="%.4f"), encoding="utf-8")
    print(output)


def synthetic(args):
    if args.n < 1 or args.length < 3 or args.features < 1:
        raise ValueError("n/features must be positive and length must be at least 3")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **generate_synthetic(args.n, args.length, args.features, seed=args.seed)
    )
    print(path)


def evaluate(args):
    """Display saved held-out results without rerunning model selection."""
    path = Path(args.run_dir) / "metrics.json"
    print(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))


def predict(args):
    """Predict causal commission risk (detection risk for SAFE-r) from a checkpoint."""
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    # Prediction requires only covariates and lengths, not labels or latent onset.
    with np.load(args.data, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        lengths = np.asarray(data["lengths"])
        ids = data["entity_id"] if "entity_id" in data else np.arange(len(x))
    if x.ndim != 3 or len(x) == 0 or lengths.shape != (len(x),):
        raise ValueError("Expected nonempty x[N,L,D] and lengths[N]")
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("lengths must contain integers")
    if np.any(lengths < 1) or np.any(lengths > x.shape[1]):
        raise ValueError("lengths outside padded sequence range")
    options = dict(cfg["model"])
    name = options.pop("name")
    model = build_model(name, x.shape[-1], **options).to(args.device)
    model.load_state_dict(
        torch.load(run_dir / "model.pt", map_location=args.device, weights_only=True)
    )
    model.eval()
    risks = []
    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            xb = torch.as_tensor(x[start : start + args.batch_size], device=args.device)
            lb = torch.as_tensor(
                lengths[start : start + args.batch_size], dtype=torch.long, device=args.device
            )
            risk = model(xb, lb)["risk"]
            final = risk.gather(1, (lb - 1)[:, None])
            valid = torch.arange(x.shape[1], device=args.device)[None] < lb[:, None]
            risks.append(torch.where(valid, risk, final).cpu().numpy())
    risk = np.concatenate(risks)
    threshold = json.loads((run_dir / "metrics.json").read_text())["threshold"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        risk=risk,
        final_risk=risk[:, -1],
        prediction=risk[:, -1] >= threshold,
        threshold=threshold,
        lengths=lengths,
        entity_id=ids,
    )
    print(output)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="safeplus", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    cmd = commands.add_parser("synthetic", help="Generate a synthetic NPZ dataset")
    cmd.add_argument("--output", default="data/processed/synthetic_demo.npz")
    cmd.add_argument("--n", type=int, default=2000)
    cmd.add_argument("--length", type=int, default=24)
    cmd.add_argument("--features", type=int, default=8)
    cmd.add_argument("--seed", type=int, default=0)
    cmd.set_defaults(func=synthetic)

    cmd = commands.add_parser("train", help="Train and save a model plus held-out metrics")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--epochs", type=int)
    cmd.add_argument("--batch-size", type=int)
    cmd.add_argument("--device", help="PyTorch device, e.g. cpu, cuda, mps")
    cmd.add_argument("--seed", type=int)
    cmd.add_argument("--output-dir", help="Parent directory; outputs go into seed_<seed>")
    cmd.set_defaults(func=train)

    cmd = commands.add_parser("evaluate", help="Display saved held-out metrics")
    cmd.add_argument("--run-dir", required=True)
    cmd.set_defaults(func=evaluate)

    cmd = commands.add_parser("predict", help="Run a saved model on an NPZ dataset")
    cmd.add_argument("--run-dir", required=True)
    cmd.add_argument("--data", required=True, help="NPZ containing x and lengths")
    cmd.add_argument("--output", required=True, help="Output predictions NPZ")
    cmd.add_argument("--batch-size", type=int, default=64)
    cmd.add_argument("--device", default="cpu")
    cmd.set_defaults(func=predict)

    cmd = commands.add_parser("aggregate", help="Summarize runs as CSV and LaTeX")
    cmd.add_argument("--runs", default="runs")
    cmd.add_argument("--output", default="reports")
    cmd.set_defaults(func=aggregate)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
