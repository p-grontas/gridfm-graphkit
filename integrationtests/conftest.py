import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--calibrate",
        type=int,
        default=0,
        help="Run training N times to collect metric mean/std for range calibration. "
        "Skips metric range assertions. Example: pytest --calibrate 5",
    )


@pytest.fixture
def calibrate_runs(request):
    """Number of calibration runs requested via --calibrate (0 = normal test mode)."""
    return request.config.getoption("--calibrate")

