from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from functools import partial
from gridfm_graphkit.datasets.rrwp import add_full_rrwp

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

def compute_posenc_stats(data, pe_types, cfg):
    """Precompute positional encodings for the given graph.
    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'RRWP': Relative Random Walk Probabilities PE (Ours, for GRIT)
    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet',
                     'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE','RRWP']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    if 'RRWP' in pe_types:
        param = cfg.posenc_RRWP
        transform = partial(add_full_rrwp,
                            walk_length=param.ksteps,
                            attr_name_abs="rrwp",
                            attr_name_rel="rrwp",
                            add_identity=True
                            )
        data = transform(data)

    return data


class ComputePosencStat(BaseTransform):
    def __init__(self, pe_types, cfg):
        self.pe_types = pe_types
        self.cfg = cfg

    def __call__(self, data: Data) -> Data:
        data = compute_posenc_stats(data, 
                                    pe_types=self.pe_types,
                                    cfg=self.cfg
                                    )
        return data