import importlib as _importlib

__all__ = ["PowerFlowTask", "OptimalPowerFlowTask", "StateEstimationTask"]

_LAZY_IMPORTS = {
    "PowerFlowTask": ("gridfm_graphkit.tasks.pf_task", "PowerFlowTask"),
    "OptimalPowerFlowTask": ("gridfm_graphkit.tasks.opf_task", "OptimalPowerFlowTask"),
    "StateEstimationTask": ("gridfm_graphkit.tasks.se_task", "StateEstimationTask"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
