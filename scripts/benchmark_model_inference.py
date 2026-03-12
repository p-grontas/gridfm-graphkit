#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A unified script for benchmarking and limited custom profiling. Benchmarking columns in the output csv are [batch_size,avg_time_per_sample_ms].

Supports two model types via --model flag:
  - "hetero" (default): GNS_heterogeneous with HeteroData (bus + gen nodes)
  - "grit": GritTransformer with homogeneous Data (single node type)

Example usage — Heterogeneous GNS (edge count is 2*E (branch count)):

######################################

CONF_PATH=../examples/config
OUT_DIR=../scripts
mkdir $OUT_DIR

python benchmark_model_inference.py --model hetero --config $CONF_PATH/case30_ieee_base.yaml --num_nodes 30 --num_edges 82 --num_gens 6 --iterations 20 --output_csv $OUT_DIR/case30.csv || true
python benchmark_model_inference.py --model hetero --config $CONF_PATH/case118_ieee_base.yaml --num_nodes 118 --num_edges 372 --num_gens 54 --iterations 20 --output_csv $OUT_DIR/case118.csv || true

######################################

Example usage — GRIT (homogeneous, --num_gens is ignored):

######################################

python benchmark_model_inference.py --model grit --config $CONF_PATH/grit_pretraining.yaml --num_nodes 30 --num_edges 82 --iterations 20 --output_csv $OUT_DIR/grit_case30.csv || true
python benchmark_model_inference.py --model grit --config $CONF_PATH/grit_pretraining.yaml --num_nodes 118 --num_edges 372 --iterations 20 --output_csv $OUT_DIR/grit_case118.csv || true

######################################

Author(s): Mangaliso M. - mngomezulum@ibm.com
           Matteo M. - Not Available
"""

import os
import time
import csv
import yaml
import torch
import argparse
import platform
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, HeteroData
from gridfm_graphkit.io.param_handler import NestedNamespace, load_model

# Optional: tqdm (imported but not required for core flow)
try:
    from tqdm import tqdm  # noqa: F401
except Exception:
    pass

# Compilation (kept from original)
import torch._dynamo as dynamo
dynamo.config.suppress_errors = False

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser(description="Benchmark GNN Model inference with profiling CSV")
parser.add_argument("--model", type=str, choices=["hetero", "grit"], default="hetero",
                    help="Model type: 'hetero' for GNS_heterogeneous, 'grit' for GritTransformer")
parser.add_argument("--config", type=str, required=True, help="Path to config YAML for model")
parser.add_argument("--num_nodes", type=int, required=True)
parser.add_argument("--num_gens", type=int, default=0,
                    help="Number of generator nodes (required for hetero, ignored for grit)")
parser.add_argument("--num_edges", type=int, required=True)
parser.add_argument("--output_csv", type=str, required=True)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory in DataLoader when CUDA is available")
args = parser.parse_args()

# --- Custom logging (ensure directory exists)
import logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger('ibm_benchmark_logger')
logger.setLevel(logging.DEBUG)
logger.propagate = False
file_handler = logging.FileHandler('logs/ibm_bench_logs.log', mode='a')  # 'a' for append, 'w' to overwrite
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)

# ----------------------------
# Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.config, "r") as f:
    base_config = yaml.safe_load(f)

config_args = NestedNamespace(**base_config)
model = load_model(config_args).to(device).eval()

# ----------------------------
# Parameters
# ----------------------------
MODEL_TYPE = args.model
N_BUS = args.num_nodes
N_GEN = args.num_gens
E = args.num_edges
EDGE_FEATS = config_args.model.edge_dim

if MODEL_TYPE == "hetero":
    BUS_FEATS = config_args.model.input_bus_dim
    GEN_FEATS = config_args.model.input_gen_dim
    NODE_FEATS = None  # not used for hetero
else:
    # GRIT homogeneous model
    NODE_FEATS = config_args.model.input_dim
    OUTPUT_DIM = config_args.model.output_dim
    MASK_DIM = getattr(config_args.data, "mask_dim", 6)
    # Positional encoding config
    RRWP_ENABLED = getattr(config_args.data.posenc_RRWP, "enable", False) if hasattr(config_args.data, "posenc_RRWP") else False
    RRWP_KSTEPS = getattr(config_args.data.posenc_RRWP, "ksteps", 21) if RRWP_ENABLED else 0
    RWSE_ENABLED = hasattr(config_args.model, "encoder") and getattr(config_args.model.encoder, "node_encoder", False) \
                   and "RWSE" in getattr(config_args.model.encoder, "node_encoder_name", "")
    RWSE_TIMES = getattr(config_args.model.encoder.posenc_RWSE.kernel, "times", 21) if RWSE_ENABLED else 0
    BUS_FEATS = NODE_FEATS  # alias for CSV output compatibility
    GEN_FEATS = 0

# Keep original batch sizes list
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 640, 768, 1024, 2048, 2560, 3072, 3584, 4096, 6144, 9216, 13824, 17280, 20736, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000]
iterations = args.iterations

# ----------------------------
# Helpers
# ----------------------------
def now_ms() -> float:
    return time.perf_counter() * 1000.0

def maybe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_env_info():
    # CPU name detection
    cpu_name = None
    try:
        cpu_name = platform.processor() or None
        if not cpu_name and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.strip().split(":", 1)[1].strip()
                        break
        if not cpu_name:
            cpu_name = platform.uname().machine
    except Exception:
        cpu_name = "unknown"

    # GPU names and device info
    if torch.cuda.is_available():
        try:
            gpu_names_list = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            gpu_names = "; ".join(gpu_names_list)
        except Exception:
            gpu_names = "cuda_available_but_name_unreadable"
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cuda"
        cuda_version_in_torch = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    else:
        # Apple Metal backend?
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple MPS"
            gpu_names = "mps"
            cuda_version_in_torch = None
            cudnn_version = None
        else:
            device_type = "cpu"
            device_name = "cpu"
            gpu_names = "none"
            cuda_version_in_torch = None
            cudnn_version = None

    info = {
        "device_type": device_type,
        "device_name": device_name,
        "gpu_names": gpu_names,
        "cpu_name": cpu_name,
        "torch_version": torch.__version__,
        "cuda_version_in_torch": cuda_version_in_torch,
        "cudnn_version": cudnn_version,
        "python_version": platform.python_version(),
    }
    return info

# ----------------------------
# Generate Synthetic Hetero Graph
# ----------------------------
def generate_hetero_graph():
    """
    Generates a dummy heterogeneous power network graph for benchmarking.

    Returns:
        data (HeteroData): single self-contained heterogeneous graph with:
            - data["bus"].x, data["gen"].x
            - edge_index & edge_attr for all relations
            - mask_dict inside data.mask_dict
    """
    data = HeteroData()

    # Node features
    data["bus"].x = torch.randn(N_BUS, BUS_FEATS)
    data["gen"].x = torch.randn(N_GEN, GEN_FEATS)

    # Edges: Bus–Bus
    src = torch.randint(0, N_BUS, (E,))
    dst = torch.randint(0, N_BUS, (E,))
    data["bus", "connects", "bus"].edge_index = torch.stack([src, dst], dim=0)
    data["bus", "connects", "bus"].edge_attr = torch.randn(E, EDGE_FEATS)

    # Edges: Gen–Bus & Bus–Gen
    gen_to_bus = torch.randint(0, N_BUS, (N_GEN,))

    # Gen → Bus
    data["gen", "connected_to", "bus"].edge_index = torch.stack(
        [torch.arange(N_GEN), gen_to_bus], dim=0
    )

    # Bus → Gen
    data["bus", "connected_to", "gen"].edge_index = torch.stack(
        [gen_to_bus, torch.arange(N_GEN)], dim=0
    )

    # No edge features for these
    data["gen", "connected_to", "bus"].edge_attr = None
    data["bus", "connected_to", "gen"].edge_attr = None

    # Dummy masks (all True)
    mask_bus = torch.ones_like(data["bus"].x, dtype=torch.bool)
    mask_gen = torch.ones_like(data["gen"].x, dtype=torch.bool)
    bus_types = torch.randint(0, 3, (N_BUS,))
    mask_branch = torch.ones_like(data["bus", "connects", "bus"].edge_attr, dtype=torch.bool)

    mask_PQ = bus_types == 0
    mask_PV = bus_types == 1
    mask_REF = bus_types == 2

    data.mask_dict = {
        "bus": mask_bus,
        "gen": mask_gen,
        "PQ": mask_PQ,
        "PV": mask_PV,
        "REF": mask_REF,
        "branch": mask_branch
    }
    return data


# ----------------------------
# Generate Synthetic Homogeneous Graph (GRIT)
# ----------------------------
def generate_homo_graph():
    """
    Generates a dummy homogeneous power network graph for GRIT benchmarking.

    Returns:
        data (Data): single self-contained homogeneous graph with:
            - data.x: node features [N_BUS, NODE_FEATS]
            - data.y: target labels [N_BUS, OUTPUT_DIM]
            - data.edge_index: [2, E]
            - data.edge_attr: [E, EDGE_FEATS]
            - data.pestat_RWSE (if RWSE enabled): [N_BUS, RWSE_TIMES]
            - data.rrwp, rrwp_index, rrwp_val (if RRWP enabled)
    """
    data = Data()

    # Node features: same layout as powergrid_dataset (Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF)
    data.x = torch.randn(N_BUS, NODE_FEATS)
    data.y = data.x[:, :OUTPUT_DIM].clone()

    # Edges
    src = torch.randint(0, N_BUS, (E,))
    dst = torch.randint(0, N_BUS, (E,))
    data.edge_index = torch.stack([src, dst], dim=0)
    data.edge_attr = torch.randn(E, EDGE_FEATS)

    # RWSE positional encoding (diagonal of random-walk matrix powers)
    if RWSE_ENABLED:
        data.pestat_RWSE = torch.randn(N_BUS, RWSE_TIMES).abs()

    # RRWP positional / structural encoding
    if RRWP_ENABLED:
        data.rrwp = torch.randn(N_BUS, RRWP_KSTEPS)
        # Sparse RRWP for edges: include existing edges + self-loops
        self_loops = torch.arange(N_BUS).unsqueeze(0).repeat(2, 1)
        rrwp_idx = torch.cat([data.edge_index, self_loops], dim=1)
        rrwp_nnz = rrwp_idx.size(1)
        data.rrwp_index = rrwp_idx
        data.rrwp_val = torch.randn(rrwp_nnz, RRWP_KSTEPS)

    return data

# ----------------------------
# Benchmark Function
# ----------------------------
def benchmark():
    # Environment/context info (constant per run)
    env = get_env_info()
    timestamp = datetime.now().isoformat(timespec='seconds')

    # Measure synthetic graph creation
    t0 = now_ms()
    if MODEL_TYPE == "hetero":
        data = generate_hetero_graph()
    else:
        data = generate_homo_graph()
    t1 = now_ms()
    data_gen_time_ms = t1 - t0

    # Move the base graph to device (preserve original behavior)
    maybe_cuda_sync()
    t2 = now_ms()
    data = data.to(device)
    maybe_cuda_sync()
    t3 = now_ms()
    graph_to_device_time_ms = t3 - t2

    batch_sizes_used = []
    times = []

    header = [
        # Keep original first two columns
        "batch_size",
        "avg_time_per_sample_ms",

        # Execution config
        "num_iters",
        "total_samples",

        # Data/IO timing
        "data_gen_time_ms",
        "graph_to_device_time_ms",
        "clone_list_time_ms",
        "dataloader_create_time_ms",
        "dataloader_first_iter_time_ms",
        "batch_to_device_time_ms",

        # Model timing
        "warmup_time_ms",
        "iter_total_wall_time_ms",
        "iter_gpu_time_ms",
        "gpu_idle_time_ms",
        "gpu_busy_ratio",
        "samples_per_sec_wall",
        "samples_per_sec_gpu",
        "timing_source",  # "cuda_event" or "wall_clock"

        # Memory
        "max_cuda_mem_alloc_bytes",
        "max_cuda_mem_reserved_bytes",

        # Graph & model context
        "n_bus", "n_gen", "n_edges",
        "bus_feats", "gen_feats", "edge_feats",

        # Runtime context
        "device_type", "device_name",
        "torch_version", "cuda_version_in_torch", "cudnn_version",
        "python_version",
        "cpu_name",             # NEW
        "gpu_names",            # NEW
        "timestamp_iso",
        "num_workers",
        "pin_memory",
    ]

    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for batch_size in batch_sizes:
            # Build list of graphs (on device, preserving original flow)
            maybe_cuda_sync()
            t_clone_start = now_ms()
            data_list = [data.clone() for _ in range(batch_size)]
            maybe_cuda_sync()
            t_clone_end = now_ms()
            clone_list_time_ms = t_clone_end - t_clone_start

            # Create DataLoader
            pin_mem = args.pin_memory and torch.cuda.is_available()
            persistent = args.num_workers > 0
            t_dl_create_start = now_ms()
            loader = DataLoader(
                data_list,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_mem,
                persistent_workers=persistent,
            )
            t_dl_create_end = now_ms()
            dataloader_create_time_ms = t_dl_create_end - t_dl_create_start

            # Fetch first batch (collate)
            t_iter_start = now_ms()
            batch = next(iter(loader))
            t_iter_end = now_ms()
            dataloader_first_iter_time_ms = t_iter_end - t_iter_start

            # Ensure batch on device (likely ~0 if items already on device)
            maybe_cuda_sync()
            t_b2d_start = now_ms()
            batch = batch.to(device, non_blocking=True) if torch.cuda.is_available() else batch.to(device)
            maybe_cuda_sync()
            t_b2d_end = now_ms()
            batch_to_device_time_ms = t_b2d_end - t_b2d_start

            test_model = model

            # Warmup (excluded from main timing)
            maybe_cuda_sync()
            t_warmup_start = now_ms()
            with torch.no_grad():
                for _ in range(5):
                    if MODEL_TYPE == "hetero":
                        _ = test_model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
                    else:
                        _ = test_model(batch.clone())
            maybe_cuda_sync()
            t_warmup_end = now_ms()
            warmup_time_ms = t_warmup_end - t_warmup_start

            num_iters = iterations
            total_samples = batch_size * num_iters

            # Reset CUDA memory stats and set up timing
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

            # Iteration timing
            maybe_cuda_sync()
            wall_start = now_ms()
            with torch.no_grad():
                if torch.cuda.is_available():
                    start_event.record()
                for _ in range(num_iters):
                    if MODEL_TYPE == "hetero":
                        _ = test_model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
                    else:
                        _ = test_model(batch.clone())
                if torch.cuda.is_available():
                    end_event.record()
            maybe_cuda_sync()
            wall_end = now_ms()

            iter_total_wall_time_ms = wall_end - wall_start

            if torch.cuda.is_available():
                iter_gpu_time_ms = float(start_event.elapsed_time(end_event))  # ms
                timing_source = "cuda_event"
                avg_time_per_sample_ms = iter_gpu_time_ms / total_samples
                gpu_idle_time_ms = max(iter_total_wall_time_ms - iter_gpu_time_ms, 0.0)
                gpu_busy_ratio = (iter_gpu_time_ms / iter_total_wall_time_ms) if iter_total_wall_time_ms > 0 else None
                max_cuda_mem_alloc_bytes = int(torch.cuda.max_memory_allocated(device))
                max_cuda_mem_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
                samples_per_sec_gpu = (total_samples / (iter_gpu_time_ms / 1000.0)) if iter_gpu_time_ms > 0 else None
            else:
                iter_gpu_time_ms = None
                timing_source = "wall_clock"
                avg_time_per_sample_ms = iter_total_wall_time_ms / total_samples
                gpu_idle_time_ms = None
                gpu_busy_ratio = None
                max_cuda_mem_alloc_bytes = None
                max_cuda_mem_reserved_bytes = None
                samples_per_sec_gpu = None

            samples_per_sec_wall = (total_samples / (iter_total_wall_time_ms / 1000.0)) if iter_total_wall_time_ms > 0 else None

            # Prepare row
            row = [
                batch_size,
                avg_time_per_sample_ms,

                num_iters,
                total_samples,

                data_gen_time_ms,
                graph_to_device_time_ms,
                clone_list_time_ms,
                dataloader_create_time_ms,
                dataloader_first_iter_time_ms,
                batch_to_device_time_ms,

                warmup_time_ms,
                iter_total_wall_time_ms,
                iter_gpu_time_ms,
                gpu_idle_time_ms,
                gpu_busy_ratio,
                samples_per_sec_wall,
                samples_per_sec_gpu,
                timing_source,

                max_cuda_mem_alloc_bytes,
                max_cuda_mem_reserved_bytes,

                N_BUS, N_GEN, E,
                BUS_FEATS, GEN_FEATS, EDGE_FEATS,

                env["device_type"], env["device_name"],
                env["torch_version"], env["cuda_version_in_torch"], env["cudnn_version"],
                env["python_version"],
                env["cpu_name"],
                env["gpu_names"],
                timestamp,
                args.num_workers,
                bool(pin_mem),
            ]

            writer.writerow(row)
            csvfile.flush()
            batch_sizes_used.append(batch_size)
            times.append(avg_time_per_sample_ms)

    return batch_sizes_used, times


if __name__ == "__main__":
    print(f"Starting benchmark for {os.path.basename(args.output_csv)} ..")
    benchmark()
    print(f"Finished benchmarking for {os.path.basename(args.output_csv)}\n ...")
