from pathlib import Path

DATASET_ID = "nvidia/LIBERO_LeRobot_v3"
BASE_MODEL = "lerobot/smolvla_base"

TARGETS = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
}
BUDGETS = (5, 10, 25)
TARGET_SEEDS = (42, 123)

DATA = Path("data")
RAW = DATA / "raw"
FIXED = DATA / "fixed"
SUBSETS = DATA / "subsets"
SEEN_TRAIN_ROOT = FIXED / "libero_90"
SEEN_REPO_ID = "local/libero_90_fixed"

OUTPUTS = Path("outputs")
RESULTS = Path("results")
EXTERNAL = Path("_external")
