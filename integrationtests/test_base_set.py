import pytest
import subprocess
import os
import glob
import pandas as pd
import yaml
import shutil
import zipfile
import gdown
import tempfile


def execute_and_live_output(cmd) -> None:
    subprocess.run(cmd, text=True, shell=True, check=True)


def prepare_training_config():
    """
    Modify the PF training config to set epochs to 2 for testing.
    """
    config_path = "examples/config/HGNS_PF_datakit_case14.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        config["training"] = {}

    config["training"]["epochs"] = 2

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Training config updated: epochs set to {config['training']['epochs']}")

    return config_path


def prepare_opf_training_config():
    """
    Modify the OPF training config to set epochs to 2 for testing.
    """
    config_path = "examples/config/HGNS_OPF_datakit_case14.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        config["training"] = {}

    config["training"]["epochs"] = 2

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"OPF training config updated: epochs set to {config['training']['epochs']}")

    return config_path


@pytest.fixture
def cleanup_test_artifacts():
    """
    Backup modified files and remove generated artifacts after the test.
    """
    training_config = " "
    backup_config = training_config + ".bak"

    if os.path.exists(training_config):
        shutil.copy2(training_config, backup_config)

    yield

    # Restore training config
    if os.path.exists(backup_config):
        shutil.move(backup_config, training_config)

    # Remove downloaded config
    config_file = "integrationtests/default.yaml"
    if os.path.exists(config_file):
        os.remove(config_file)

    # Remove generated directories
    for d in ["data_out", "logs"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def test_train(cleanup_test_artifacts):
    """
    Integration test for gridfm-datakit data generation and gridfm-graphkit training.

    Steps:
    1. Generate power grid data using gridfm-datakit
    2. Train a model using gridfm-graphkit
    3. Validate the PBE Mean metric
    """

    data_dir = "data_out"

    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Data directory not found or empty, downloading pre-generated data...")

        gdrive_file_id = "1NtE_4Fn3-1_BNWidZVFeSTfXf3-B50Yr"
        zip_filename = "case14_ieee.10000_scenarios_2_variants.zip"
        gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        print(f"Downloading {zip_filename} from Google Drive...")
        gdown.download(gdrive_url, zip_filename, quiet=False)

        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_filename, "r") as zf:
            zf.extractall(".")

        os.remove(zip_filename)
        print(f"Data extracted to '{data_dir}'.")
    else:
        print(f"Data directory '{data_dir}' already exists, skipping download.")

    training_config_path = prepare_training_config()

    execute_and_live_output(
        f"gridfm_graphkit train "
        f"--config {training_config_path} "
        f"--data_path data_out/ "
        f"--exp_name exp1 "
        f"--run_name run1 "
        f"--log_dir logs",
    )

    log_base = "logs"

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


@pytest.fixture
def cleanup_opf_test_artifacts():
    """
    Remove generated artifacts after the OPF test.
    """
    yield

    for d in ["data_out_opf", "logs_opf"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def test_train_opf(cleanup_opf_test_artifacts):
    """
    Integration test for OPF data download and gridfm-graphkit OPF training.

    Steps:
    1. Download pre-generated OPF power grid data from Google Drive
    2. Train a model using gridfm-graphkit with the OPF config
    3. Validate OPF-specific metrics
    """

    opf_data_dir = "data_out_opf"

    if not os.path.exists(opf_data_dir) or not os.listdir(opf_data_dir):
        print("OPF data directory not found or empty, downloading pre-generated data...")

        gdrive_file_id = "1Ow4SYGAYQ4mZad4yNKYmXvG24BbPs8Pj"
        zip_filename = "case14_ieee.10000_scenarios_2_variants_opf.zip"
        gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        print(f"Downloading {zip_filename} from Google Drive...")
        gdown.download(gdrive_url, zip_filename, quiet=False)

        print(f"Extracting {zip_filename}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_filename, "r") as zf:
                zf.extractall(tmpdir)
            shutil.move(os.path.join(tmpdir, "data_out"), opf_data_dir)

        os.remove(zip_filename)
        print(f"OPF data extracted to '{opf_data_dir}'.")
    else:
        print(f"OPF data directory '{opf_data_dir}' already exists, skipping download.")

    training_config_path = prepare_opf_training_config()

    execute_and_live_output(
        f"gridfm_graphkit train "
        f"--config {training_config_path} "
        f"--data_path {opf_data_dir}/ "
        f"--exp_name exp_opf "
        f"--run_name run1 "
        f"--log_dir logs_opf",
    )

    log_base = "logs_opf"

    exp_dirs = glob.glob(os.path.join(log_base, "*"))
    assert len(exp_dirs) > 0, "No experiment directories found in logs_opf/"

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
    metrics = dict(zip(df["Metric"], df["Value"].astype(float)))

    checks = {
        "Avg. active res. (MW)": (0.0, 2.0),
        "Avg. reactive res. (MVar)": (0.0, 2.0),
        "RMSE PG generators (MW)": (0.0, 50.0),
        "Mean optimality gap (%)": (0.0, 10.0),
        "Mean branch thermal violation from (MVA)": (0.0, 5.0),
        "Mean branch thermal violation to (MVA)": (0.0, 5.0),
        "Mean branch angle difference violation (radians)": (0.0, 1.0),
        "Mean Qg violation PV buses": (0.0, 5.0),
        "Mean Qg violation REF buses": (0.0, 5.0),
        "Mean Qg violation": (0.0, 5.0),
    }

    for metric_name, (lo, hi) in checks.items():
        assert metric_name in metrics, f"Metric '{metric_name}' not found in CSV"
        value = metrics[metric_name]
        assert lo <= value <= hi, (
            f"Metric '{metric_name}' value {value} is outside acceptable range [{lo}, {hi}]"
        )
        print(f"{metric_name}: {value} is within [{lo}, {hi}]")
