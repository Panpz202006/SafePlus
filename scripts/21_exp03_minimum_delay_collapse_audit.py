from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    r"reports/experiment03/minimum_delay_consistency"
)


def audit(inference):

    path = (
        ROOT
        / inference
        / "entity_diagnostics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    fraud = df[
        (df["true_tf"] >= 0)
        &
        (df["detected"] == 1)
    ].copy()

    fraud["true_delay"] = (
        fraud["observed_td"]
        - fraud["true_tf"]
    )

    fraud["mode_delay"] = (
        fraud["observed_td"]
        - fraud["posterior_mode_tf"]
    )

    fraud["mean_delay"] = (
        fraud["observed_td"]
        - fraud["posterior_mean_tf"]
    )

    print("=" * 80)
    print(inference)
    print("=" * 80)

    print(
        "Detected fraud users:",
        len(fraud),
    )

    print()

    print("TRUE DELAY")
    print(
        "  mean:",
        fraud["true_delay"].mean(),
    )
    print(
        "  median:",
        fraud["true_delay"].median(),
    )

    print()

    print("PREDICTED DELAY")
    print(
        "  mode mean:",
        fraud["mode_delay"].mean(),
    )
    print(
        "  mode median:",
        fraud["mode_delay"].median(),
    )
    print(
        "  posterior mean delay:",
        fraud["mean_delay"].mean(),
    )

    print()

    mode_eq_td_minus_1 = (
        fraud["mode_delay"] == 1
    ).mean()

    mode_le_1 = (
        fraud["mode_delay"] <= 1
    ).mean()

    mode_le_2 = (
        fraud["mode_delay"] <= 2
    ).mean()

    print("COLLAPSE CHECK")
    print(
        "  mode == td - 1 rate:",
        mode_eq_td_minus_1,
    )
    print(
        "  predicted delay <= 1 rate:",
        mode_le_1,
    )
    print(
        "  predicted delay <= 2 rate:",
        mode_le_2,
    )

    print()

    print(
        "True delay distribution:"
    )

    print(
        fraud[
            "true_delay"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Predicted mode delay distribution:"
    )

    print(
        fraud[
            "mode_delay"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    columns = [
        "entity_id",
        "true_tf",
        "observed_td",
        "true_delay",
        "posterior_mode_tf",
        "mode_delay",
        "posterior_mean_tf",
        "mean_delay",
    ]

    print(
        fraud[
            columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    output = (
        ROOT
        / inference
        / "minimum_delay_collapse_audit.csv"
    )

    fraud.to_csv(
        output,
        index=False,
    )

    print()
    print("Saved:", output)
    print()


def main():

    print(
        "EXPERIMENT 03N:"
    )

    print(
        "Minimum-delay collapse audit"
    )

    print()

    print(
        "核心问题："
    )

    print(
        "禁止 tf = td 以后，"
    )

    print(
        "模型是否又大量选择 tf = td - 1？"
    )

    print()

    audit(
        "exact"
    )

    audit(
        "stochastic_em"
    )


if __name__ == "__main__":
    main()