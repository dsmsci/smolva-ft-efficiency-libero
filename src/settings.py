from pathlib import Path

DATASET_ID = "nvidia/LIBERO_LeRobot_v3"
BASE_MODEL = "lerobot/smolvla_base"

# Names expected by the pretrained SmolVLA checkpoint.
DATASET_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.wrist_image": "observation.images.camera2",
}
ENV_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

TARGETS = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
}
BUDGETS = (5, 10, 25)
TARGET_SEEDS = (42, 123)

# Decoding is done directly from the original dataset videos. No mandatory
# AV1 -> H264 conversion is performed by this repository.
VIDEO_BACKEND = "torchcodec"

DATA = Path("data")
RAW = DATA / "raw"
FIXED = DATA / "fixed"
VIEWS = DATA / "views"
SEEN_TRAIN_ROOT = FIXED / "libero_90"
GOAL_TRAIN_ROOT = FIXED / "libero_goal"
SEEN_REPO_ID = "local/libero_90_fixed"
GOAL_REPO_ID = "local/libero_goal_fixed"

OUTPUTS = Path("outputs")
RESULTS = Path("results")
