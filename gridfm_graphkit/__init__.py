import importlib as _importlib

__all__ = [
    "datasets",
    "tasks",
    "models",
]

_LAZY_SUBMODULES = {
    "datasets": "gridfm_graphkit.datasets",
    "tasks": "gridfm_graphkit.tasks",
    "models": "gridfm_graphkit.models",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return _importlib.import_module(_LAZY_SUBMODULES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
