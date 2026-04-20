import pytest
import subprocess
import os
import glob
import pandas as pd
import yaml
import urllib.request
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONTESTS_DIR = REPO_ROOT / "integrationtests"
DATA_CACHE_DIR = INTEGRATIONTESTS_DIR / "data_out"
LOG_DIR = INTEGRATIONTESTS_DIR / "logs"
GENERATED_DATAKIT_CONFIG = INTEGRATIONTESTS_DIR / "_generated_default.yaml"
GENERATED_TRAINING_CONFIG = (
    INTEGRATIONTESTS_DIR / "_generated_HGNS_PF_datakit_case14.yaml"
)


def execute_and_live_output(cmd) -> None:
    subprocess.run(cmd, text=True, shell=True, check=True, cwd=REPO_ROOT)


def cached_dataset_exists(data_dir: Path) -> bool:
    raw_dir = data_dir / "case14_ieee" / "raw"
    required_files = (
        "bus_data.parquet",
        "gen_data.parquet",
        "branch_data.parquet",
    )
    return raw_dir.is_dir() and all((raw_dir / filename).exists() for filename in required_files)


def prepare_config():
    """
    Download default.yaml from gridfm-datakit repo and modify it with test parameters.
    """
    config_url = "https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/scripts/config/default.yaml"
    config_path = GENERATED_DATAKIT_CONFIG

    print(f"Downloading config from {config_url}...")
    with urllib.request.urlopen(config_url) as response:
        config_content = response.read().decode("utf-8")

    config = yaml.safe_load(config_content)

    config["network"]["name"] = "case14_ieee"
    config["load"]["scenarios"] = 10000
    config["topology_perturbation"]["n_topology_variants"] = 2
    config["settings"]["large_chunk_size"] = 10000
    config["settings"]["data_dir"] = os.fspath(DATA_CACHE_DIR.relative_to(REPO_ROOT))

    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config prepared at {config_path} with:")
    print(f"  - network.name: {config['network']['name']}")
    print(f"  - load.scenarios: {config['load']['scenarios']}")
    print(
        f"  - topology_perturbation.n_topology_variants: "
        f"{config['topology_perturbation']['n_topology_variants']}",
    )

    return config_path


def prepare_training_config():
    """
    Modify the training config to set epochs to 2 for testing.
    """
    config_path = REPO_ROOT / "examples" / "config" / "HGNS_PF_datakit_case14.yaml"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        config["training"] = {}

    config["training"]["epochs"] = 2

    with GENERATED_TRAINING_CONFIG.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Training config updated: epochs set to {config['training']['epochs']}")

    return GENERATED_TRAINING_CONFIG


@pytest.fixture
def cleanup_test_artifacts():
    """
    Remove generated test-only files and logs after the test.
    """
    yield

    for path in (GENERATED_DATAKIT_CONFIG, GENERATED_TRAINING_CONFIG):
        if path.exists():
            path.unlink()

    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR, ignore_errors=True)


def test_train(cleanup_test_artifacts):
    """
    Integration test for gridfm-datakit data generation and gridfm-graphkit training.

    Steps:
    1. Generate power grid data using gridfm-datakit
    2. Train a model using gridfm-graphkit
    3. Validate the PBE Mean metric
    """

    data_dir = DATA_CACHE_DIR

    if not cached_dataset_exists(data_dir):
        print(f"Cached dataset not found in '{data_dir}', generating data...")

        config_path = prepare_config()

        execute_and_live_output(f"gridfm_datakit generate {config_path}")
    else:
        print(f"Reusing cached dataset from '{data_dir}'.")

    training_config_path = prepare_training_config()

    execute_and_live_output(
        f"gridfm_graphkit train "
        f"--config {training_config_path} "
        f"--data_path {data_dir}/ "
        f"--exp_name exp1 "
        f"--run_name run1 "
        f"--log_dir {LOG_DIR}/",
    )

    log_base = os.fspath(LOG_DIR)

    exp_dirs = glob.glob(os.path.join(log_base, "*"))
    assert len(exp_dirs) > 0, "No experiment directories found in logs/"

    latest_exp_dir = sorted(exp_dirs, key=os.path.getctime)[-1]

    run_dirs = glob.glob(os.path.join(latest_exp_dir, "*"))
    assert len(run_dirs) > 0, f"No run directories found in {latest_exp_dir}"

    latest_run_dir = max(run_dirs, key=os.path.getmtime)

    metrics_file = os.path.join(
        latest_run_dir,
        "artifacts",
        "test",
        "case14_ieee_metrics.csv",
    )

    assert os.path.exists(metrics_file), f"Metrics file not found: {metrics_file}"

    df = pd.read_csv(metrics_file)

    pbe_mean_row = df[df["Metric"] == "PBE Mean"]
    assert len(pbe_mean_row) > 0, "PBE Mean metric not found in CSV"

    pbe_mean_value = float(pbe_mean_row.iloc[0]["Value"])

    assert 1.1 <= pbe_mean_value <= 2.9, (
        f"PBE Mean value {pbe_mean_value} is outside acceptable range [1.1, 2.9]"
    )

    print(f"PBE Mean value {pbe_mean_value} is within acceptable range [1.1, 2.9]")
