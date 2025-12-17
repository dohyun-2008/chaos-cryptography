"""Central configuration for Chaos-TPM experiments."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

TPM_PARAMS = {
    "K": 3,
    "N": 8,
    "L": 3,
}

CHAOS_PARAMS = {
    "logistic_r": 3.99,
    "tent_r": 1.99,
    "default_x0": 0.5,
}

NUM_SEEDS = 10
SEED_BASE = 1234
MAX_SYNC_STEPS = 50000

QUICK_TEST_MODE = False
QUICK_TEST_SEEDS = 3
QUICK_TEST_MODES = ["logistic"]
QUICK_TEST_MAX_SYNC_STEPS = 10000

LEARNING_RULE = "hebbian"

IMAGE_PATH = PROJECT_ROOT / "data" / "sample.png"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
CSV_DIR = RESULTS_DIR / "csv"
IMAGES_DIR = RESULTS_DIR / "images"

CHAOS_MODES = ["logistic", "tent", "interleaved", "cascade"]

CORRELATION_SAMPLES = 5000

DEBUG_CHAOS_VALUES = True
DEBUG_SAMPLE_SIZE = 50

LOG_LEVEL = "INFO"
LOG_INDIVIDUAL_EXPERIMENTS = False

PLOT_FORMATS = ["png", "pdf"]
PLOT_DPI = 300

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
