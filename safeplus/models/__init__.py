from .baselines import (
    GRUClassifier,
    LSTMClassifier,
    LSTMSAFER,
    SAFER,
)

from .safe import (
    SAFE,
    LSTMSAFE,
)

from .safeplus import SafePlus


def build_model(
    name: str,
    input_dim: int,
    hidden_dim: int,
    **kwargs,
):
    models = {
        "gru": GRUClassifier,
        "lstm": LSTMClassifier,

        # Detection-time survival baselines
        "safe-r": SAFER,
        "lstm-r": LSTMSAFER,

        # Fraud-onset-only baselines
        "safe": SAFE,
        "lstm-safe": LSTMSAFE,

        # Joint model
        "safeplus": SafePlus,
    }

    if name not in models:
        raise ValueError(
            f"Unknown model {name!r}; "
            f"choose from {sorted(models)}"
        )

    return models[name](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        **kwargs,
    )


__all__ = [
    "SAFE",
    "LSTMSAFE",
    "SAFER",
    "GRUClassifier",
    "LSTMClassifier",
    "LSTMSAFER",
    "SafePlus",
    "build_model",
]