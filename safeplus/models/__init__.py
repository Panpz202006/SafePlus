from .baselines import SAFER, GRUClassifier
from .safe import SAFE
from .safeplus import SafePlus


def build_model(name: str, input_dim: int, hidden_dim: int, **kwargs):
    models = {
        "gru": GRUClassifier,
        "safe": SAFE,
        "safe-r": SAFER,
        "safeplus": SafePlus,
    }
    if name not in models:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(models)}")
    return models[name](input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)


__all__ = ["SAFE", "SAFER", "GRUClassifier", "SafePlus", "build_model"]
