# ============================================
# File: run_pipeline.py
# ============================================

import argparse
import time
from typing import List, Tuple

from shared.config import COMPLEXITIES
from pretrain.main import run_batch_experiment as run_pretrain_batch
from pretrain.gauges import run_gauge_analysis
from mirrors.main import run_batch_mirrors


def main():
    parser = argparse.ArgumentParser(
        description="Run pretraining, gauge analysis, and mirrors adaptation."
    )

    # Parallelization Arguments
    parser.add_argument("--hs", type=int, default=None, help="Hidden Size override")
    parser.add_argument("--layers", type=int, default=None, help="Layers override")

    # Phase Switches
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretraining.")
    parser.add_argument("--skip-gauges", action="store_true", help="Skip gauge analysis.")
    parser.add_argument("--skip-mirrors", action="store_true", help="Skip mirrors adaptation.")
    parser.add_argument(
        "--include-unsolved-mirrors",
        action="store_true",
        help="Run mirrors for agents that did not reach PRETRAIN_TARGET_RETURN.",
    )

    args = parser.parse_args()

    # Determine which configs to run
    target_configs: List[Tuple[int, int]] = []
    if args.hs is not None and args.layers is not None:
        target_configs = [(args.hs, args.layers)]
        print(f"Running SINGLE process mode: HS={args.hs}, Layers={args.layers}")
    else:
        target_configs = COMPLEXITIES
        print(f"Running BATCH mode: {len(target_configs)} configurations.")

    t0 = time.time()

    # -------------------------
    # 1. Pretrain
    # -------------------------
    if not args.skip_pretrain:
        print("\n==============================")
        print("  PHASE 1: PRETRAINING")
        print("==============================")
        run_pretrain_batch(target_configs)
    else:
        print("\n[Pipeline] Skipping pretraining phase (--skip-pretrain).")

    # -------------------------
    # 2. Gauge identification
    # -------------------------
    if not args.skip_gauges:
        print("\n==============================")
        print("  PHASE 2: GAUGE IDENTIFICATION")
        print("==============================")
        run_gauge_analysis(target_configs)
    else:
        print("\n[Pipeline] Skipping gauge phase (--skip-gauges).")

    # -------------------------
    # 3. Mirrors adaptation
    # -------------------------
    if not args.skip_mirrors:
        print("\n==============================")
        print("  PHASE 3: HALL-OF-MIRRORS ADAPTATION")
        print("==============================")
        run_batch_mirrors(
            include_unsolved=args.include_unsolved_mirrors,
            target_configs=target_configs
        )
    else:
        print("\n[Pipeline] Skipping mirrors phase (--skip-mirrors).")

    t1 = time.time()
    print("\n==============================")
    print("  PIPELINE COMPLETE")
    print("==============================")
    print(f"Total wall-clock time: {t1 - t0:.1f} seconds.")


if __name__ == "__main__":
    main()