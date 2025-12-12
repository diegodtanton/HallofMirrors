# ============================================
# File: shared/config.py
# ============================================

from typing import List, Tuple
import os

# --------------------------
# World / Environment
# --------------------------
GRID_SIZE = 8
MAX_STEPS_PER_EPISODE = 100
FRAME_STACK_SIZE = 4
N_GOOD_TILES = 20
N_BAD_TILES = 20

# Rewards
GOOD_TILE_REWARD = 1.0
BAD_TILE_PENALTY = -0.5
WALL_PENALTY = -0.05
STEP_PENALTY = -0.01

# --------------------------
# Logging / Checkpoints
# --------------------------
PLOT_INTERVAL = 2048
CHECKPOINT_INTERVAL = 500_000 # Change for mock

# --------------------------
# Architectures
# --------------------------
COMPLEXITIES: List[Tuple[int, int]] = [
    (8, 1), (16, 1), (32, 1), (64, 1), (128, 1), (256, 1), (512, 1), (1024, 1),
    (8, 2), (16, 2), (32, 2), (64, 2), (128, 2), (256, 2), (512, 2), (1024, 2),
    (8, 3), (16, 3), (32, 3), (64, 3), (128, 3), (256, 3), (512, 3), (1024, 3),
]

# --------------------------
# Training Budgets
# --------------------------
PRETRAIN_MAX_STEPS = 6_000_000 # Change for mock
PRETRAIN_TARGET_RETURN = 12 # Change for mock

GRACE_STEPS = 100_000 # Change for mock
HALL_STEPS = 6_000_000 # Change for mock
STAGE_BUDGET = 2_000_000 # Change for mock

# --------------------------
# Paths
# --------------------------

# Uncomment for analysis
DOWNLOADED_DATA = "2025-12-09_09-59-37"
DOWNLOAD_ROOT = os.path.join("downloaded_results", DOWNLOADED_DATA)
PRETRAIN_CHECKPOINT_DIR = os.path.join(DOWNLOAD_ROOT, "pretrain", "checkpoints")
PRETRAIN_DATA_DIR = os.path.join(DOWNLOAD_ROOT, "pretrain", "data")
MIRRORS_CHECKPOINT_DIR = os.path.join(DOWNLOAD_ROOT, "mirrors", "checkpoints")
MIRRORS_DATA_DIR = os.path.join(DOWNLOAD_ROOT, "mirrors", "data")

# Uncommment for new experiment
# PRETRAIN_CHECKPOINT_DIR = os.path.join("pretrain", "checkpoints")
# PRETRAIN_DATA_DIR = os.path.join("pretrain", "data")
# MIRRORS_CHECKPOINT_DIR = os.path.join("mirrors", "checkpoints")
# MIRRORS_DATA_DIR = os.path.join("mirrors", "data")
# for d in [
#     PRETRAIN_CHECKPOINT_DIR,
#     PRETRAIN_DATA_DIR,
#     MIRRORS_CHECKPOINT_DIR,
#     MIRRORS_DATA_DIR,
# ]:
#     os.makedirs(d, exist_ok=True)