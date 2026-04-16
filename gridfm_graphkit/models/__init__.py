import importlib as _importlib

__all__ = [
    "GNS_heterogeneous",
    "GritHeteroAdapter",
    "PhysicsDecoderOPF",
    "PhysicsDecoderPF",
    "PhysicsDecoderSE",
]

_LAZY_IMPORTS = {
    "GNS_heterogeneous": ("gridfm_graphkit.models.gnn_heterogeneous_gns", "GNS_heterogeneous"),
    "GritHeteroAdapter": ("gridfm_graphkit.models.grit_transformer", "GritHeteroAdapter"),
    "PhysicsDecoderOPF": ("gridfm_graphkit.models.utils", "PhysicsDecoderOPF"),
    "PhysicsDecoderPF": ("gridfm_graphkit.models.utils", "PhysicsDecoderPF"),
    "PhysicsDecoderSE": ("gridfm_graphkit.models.utils", "PhysicsDecoderSE"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
