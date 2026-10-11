from __future__ import annotations

import importlib.util
from pathlib import Path


# ============================================================
# Experiment 03G
# Re-run the SAME latent-decoupled model on the corrected,
# model-consistent synthetic benchmark.
#
# We deliberately reuse scripts/10_exp03_latent_decoupled_joint.py
# without changing the model or training logic.
#
# Only the dataset and output locations are changed.
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/10_exp03_latent_decoupled_joint.py"
)

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Cannot find base experiment script: {BASE_SCRIPT}"
    )


# ============================================================
# Dynamically load Experiment 03F
# ============================================================

spec = importlib.util.spec_from_file_location(
    "exp03_latent_decoupled_base",
    BASE_SCRIPT,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Failed to load Experiment 03F script."
    )

exp = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    exp
)


# ============================================================
# Override ONLY the dataset and output locations.
#
# Model architecture, likelihood, optimizer, split, seed,
# batch size, hidden dimension, exact inference, stochastic EM,
# and evaluation logic all remain unchanged.
# ============================================================

exp.DATA_PATH = Path(
    r"data/processed/synthetic_consistent_delay8.npz"
)

exp.OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"latent_decoupled_joint_consistent_delay8"
)

exp.RUN_ROOT = Path(
    r"runs/synthetic/exp03/"
    r"latent-decoupled-joint-consistent-delay8"
)


# ============================================================
# Print experimental contract
# ============================================================

print(
    "=" * 80
)

print(
    "EXPERIMENT 03G: MODEL-CONSISTENT CENSORING"
)

print(
    "=" * 80
)

print(
    "Dataset:",
    exp.DATA_PATH,
)

print()

print(
    "UNCHANGED:"
)

print(
    "  x behavioral sequences"
)

print(
    "  true t_f"
)

print(
    "  entity IDs"
)

print(
    "  train/validation/test split"
)

print(
    "  GRU encoder"
)

print(
    "  occurrence/timing/detection architecture"
)

print(
    "  latent joint likelihood"
)

print(
    "  exact inference"
)

print(
    "  stochastic-EM inference"
)

print(
    "  optimizer and hyperparameters"
)

print()

print(
    "CHANGED:"
)

print(
    "  detection process / censoring only"
)

print()

print(
    "Every undetected fraud entity now satisfies:"
)

print(
    "  true t_f < observation horizon < latent t_d"
)

print()

print(
    "True t_f remains hidden from model training."
)

print(
    "It is used only for post-hoc synthetic evaluation."
)

print(
    "=" * 80
)

print()


# ============================================================
# Run the original Experiment 03F pipeline
# ============================================================

exp.main()