from __future__ import annotations

import importlib.util
from pathlib import Path


# ============================================================
# Experiment 03H
# Same-delay paired censoring control:
#
# Re-run EXACTLY the same latent-decoupled pipeline from
# Experiment 03F.
#
# ONLY dataset detection/censoring labels are changed.
#
# x, tf, candidate td, detection delay and split stay fixed.
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/10_exp03_latent_decoupled_joint.py"
)

if not BASE_SCRIPT.exists():

    raise FileNotFoundError(
        f"Cannot find base script: {BASE_SCRIPT}"
    )


# ============================================================
# Load Experiment 03F unchanged
# ============================================================


spec = importlib.util.spec_from_file_location(
    "exp03_latent_decoupled_base",
    BASE_SCRIPT,
)

if (
    spec is None
    or spec.loader is None
):

    raise RuntimeError(
        "Failed to load Experiment 03F."
    )


exp = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    exp
)


# ============================================================
# Override ONLY dataset/output paths
# ============================================================


exp.DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

exp.OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"latent_decoupled_same_delay_consistent"
)

exp.RUN_ROOT = Path(
    r"runs/synthetic/exp03/"
    r"latent-decoupled-same-delay-consistent"
)


# ============================================================
# Experimental contract
# ============================================================


print(
    "=" * 80
)

print(
    "EXPERIMENT 03H: SAME-DELAY PAIRED CENSOR CONTROL"
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
    "  behavioral sequence x"
)

print(
    "  true t_f"
)

print(
    "  detection delay"
)

print(
    "  candidate t_d"
)

print(
    "  entity IDs"
)

print(
    "  train/validation/test split"
)

print(
    "  model architecture"
)

print(
    "  latent likelihood"
)

print(
    "  exact inference"
)

print(
    "  stochastic EM"
)

print(
    "  optimizer"
)

print(
    "  hyperparameters"
)

print()

print(
    "ONLY CHANGE:"
)

print(
    "  in-window detection is no longer randomly hidden"
)

print()

print(
    "True t_f remains hidden from training."
)

print(
    "=" * 80
)

print()


# ============================================================
# Run same Experiment 03F pipeline
# ============================================================


exp.main()