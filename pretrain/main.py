# ============================================
# File: pretrain/main.py
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from shared.config import (
    COMPLEXITIES,
    PRETRAIN_MAX_STEPS,
    PRETRAIN_TARGET_RETURN,
    PRETRAIN_CHECKPOINT_DIR,
    PRETRAIN_DATA_DIR,
)
from shared.env import HallOfMirrorsGridworld, ManualFrameStack
from shared.agent import PPOAgent, PPOConfig


def run_batch_experiment(target_configs=None):
    if target_configs is None:
        target_configs = COMPLEXITIES

    for (hs, nl) in target_configs:
        run_name = f"hs{hs}_l{nl}"
        print(f"\n--- Running Pretrain Configuration: {run_name} ---")

        # 1. Setup env + agent
        # FIX: Updated to new API (random_rot, etc.)
        base_env = HallOfMirrorsGridworld(
            random_rot=False, 
            random_step=False, 
            random_val=False,
            fixed_sensor_rotation=0, 
            fixed_step_size=1, 
            fixed_good_is_red=False
        )
        env = ManualFrameStack(base_env)
        
        cfg = PPOConfig(total_steps=PRETRAIN_MAX_STEPS, lr=2.5e-4)
        
        agent = PPOAgent(env, hidden_size=hs, n_hidden_layers=nl, config=cfg)

        run_ckpt_dir = os.path.join(PRETRAIN_CHECKPOINT_DIR, run_name)
        os.makedirs(run_ckpt_dir, exist_ok=True)

        # 2. Plot callback
        def plot_cb(stats):
            if not stats:
                return
            df = pd.DataFrame(stats)
            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["instant_return"], label="Instant", alpha=0.3)
            plt.plot(df["step"], df["rolling_return"], label="Rolling (20)", linewidth=2)
            plt.xlabel("Steps")
            plt.ylabel("Average Return")
            plt.title(f"Training Progress: {run_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(run_ckpt_dir, "training_progress.png")
            plt.savefig(plot_path)
            plt.close()

            csv_path = os.path.join(PRETRAIN_DATA_DIR, f"{run_name}_progress.csv")
            df.to_csv(csv_path, index=False)

        # 3. Train
        agent.train(
            stop_at_return=PRETRAIN_TARGET_RETURN,
            checkpoint_dir=run_ckpt_dir,
            plot_callback=plot_cb,
        )


if __name__ == "__main__":
    run_batch_experiment()