import os
import tempfile
import hashlib

import torch
from torch_geometric.data import HeteroData


class CachedPosencTransform:
    """Disk-caching wrapper for positional encoding transforms.

    Computes the PE on the first access for each sample and caches the
    result to disk. Subsequent accesses load from cache, avoiding
    redundant computation across epochs and across jobs that share the
    same processed directory.

    Thread/process safety:
        - Uses atomic write (write to temp file, then os.replace) so
          concurrent DataLoader workers or separate jobs cannot produce
          corrupt cache files.
        - Cache is keyed by scenario_id, which is unique per graph.
          Different train/val/test splits across jobs safely share the
          cache since RWSE depends only on topology, not on split
          membership.

    Args:
        transform: The inner PE transform (e.g. ComputePosencStat).
        cache_dir: Directory to store cached PE tensors.
        cached_attrs: List of attribute names to cache on the bus node store
            (e.g. ["pestat_RWSE"]).
        cached_edge_type: Optional edge type tuple (e.g. ("bus", "rrwp", "bus"))
            whose edge_index and edge_attr should also be cached.
        key_attr: Attribute on the data object used as the cache key.
            Must be a scalar tensor (e.g. scenario_id).
    """

    def __init__(
        self,
        transform,
        cache_dir: str,
        cached_attrs: list[str],
        cached_edge_type: tuple[str, str, str] | None = None,
        key_attr: str = "scenario_id",
    ):
        self.transform = transform
        self.cache_dir = cache_dir
        self.cached_attrs = cached_attrs
        self.cached_edge_type = cached_edge_type
        self.key_attr = key_attr
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, data) -> str:
        key = data[self.key_attr].item()
        return os.path.join(self.cache_dir, f"pe_cache_{key}.pt")

    def _load_cache(self, cache_path, data):
        """Load cached PE attributes and attach them to data."""
        cached = torch.load(cache_path, weights_only=True)
        if isinstance(data, HeteroData):
            for attr, val in cached.items():
                if attr == "_edge_type_index":
                    data[self.cached_edge_type].edge_index = val
                elif attr == "_edge_type_attr":
                    data[self.cached_edge_type].edge_attr = val
                else:
                    data["bus"][attr] = val
        else:
            for attr, val in cached.items():
                setattr(data, attr, val)

    def _save_cache(self, cache_path, data):
        """Atomically save PE attributes to the cache file.

        Uses a temporary file in the same directory followed by
        os.replace, which is atomic on both Linux and Windows.
        This ensures concurrent workers/jobs never see a partially
        written file.
        """
        if isinstance(data, HeteroData):
            target = data["bus"]
        else:
            target = data

        to_cache = {}
        for attr in self.cached_attrs:
            if hasattr(target, attr):
                to_cache[attr] = getattr(target, attr)

        # Cache edge-type data (RRWP sparse index + values)
        if (
            self.cached_edge_type is not None
            and isinstance(data, HeteroData)
            and self.cached_edge_type in data.edge_types
        ):
            edge_store = data[self.cached_edge_type]
            if hasattr(edge_store, "edge_index"):
                to_cache["_edge_type_index"] = edge_store.edge_index
            if hasattr(edge_store, "edge_attr"):
                to_cache["_edge_type_attr"] = edge_store.edge_attr

        if not to_cache:
            return

        # Write to a temporary file in the same directory (same filesystem)
        # to guarantee os.replace is atomic.
        fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_dir,
            prefix=f".pe_cache_tmp_{os.getpid()}_",
            suffix=".pt",
        )
        try:
            os.close(fd)
            torch.save(to_cache, tmp_path)
            os.replace(tmp_path, cache_path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def __call__(self, data):
        cache_path = self._cache_path(data)

        # Fast path: load from cache if available
        if os.path.exists(cache_path):
            self._load_cache(cache_path, data)
            return data

        # Slow path: compute, then cache
        data = self.transform(data)
        self._save_cache(cache_path, data)
        return data


def make_pe_cache_dir(processed_dir: str, pe_type: str, cfg) -> str:
    """Build a cache directory path that includes a config fingerprint.

    The fingerprint ensures that changing PE parameters (e.g. kernel.times)
    invalidates the cache automatically by using a different directory.

    Args:
        processed_dir: The dataset's processed directory.
        pe_type: "RWSE" or "RRWP".
        cfg: The data config namespace containing PE parameters.

    Returns:
        Path to the cache directory.
    """
    if pe_type == "RWSE":
        kernel_times = cfg.posenc_RWSE.kernel.times
        fingerprint = f"k{kernel_times}"
    elif pe_type == "RRWP":
        ksteps = cfg.posenc_RRWP.ksteps
        topk = getattr(cfg.posenc_RRWP, "topk", 0)
        if topk and topk > 0:
            fingerprint = f"k{ksteps}_topk{topk}"
        else:
            fingerprint = f"k{ksteps}"
    else:
        fingerprint = "default"

    cache_dir_name = f"pe_cache_{pe_type.lower()}_{fingerprint}"
    return os.path.join(processed_dir, cache_dir_name)
