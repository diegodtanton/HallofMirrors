# ============================================
# File: mirrors/main.py
# ============================================

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from shared.config import (
    COMPLEXITIES,
    GRACE_STEPS,
    PRETRAIN_CHECKPOINT_DIR,
    MIRRORS_CHECKPOINT_DIR,
    MIRRORS_DATA_DIR,
    STAGE_BUDGET,
    HALL_STEPS
)
from shared.env import HallOfMirrorsGridworld, ManualFrameStack
from shared.agent import PPOAgent, PPOConfig, evaluate_agent


def run_batch_mirrors(include_unsolved: bool = False, target_configs=None):
    if target_configs is None:
        target_configs = COMPLEXITIES

    for (hs, nl) in target_configs:
        run_name = f"hs{hs}_l{nl}"
        print(f"\n=== Mirrors Adaptation for {run_name} ===")

        pretrain_dir = os.path.join(PRETRAIN_CHECKPOINT_DIR, run_name)
        if not os.path.isdir(pretrain_dir):
            print(f"  Skipping {run_name} (no pretrain dir)")
            continue

        ckpts = glob.glob(os.path.join(pretrain_dir, "*.pt"))
        if not ckpts:
            print(f"  Skipping {run_name} (no pretrain checkpoints)")
            continue

        final_ckpt = os.path.join(pretrain_dir, "final_solved.pt")
        solved = os.path.exists(final_ckpt)

        if solved:
            ckpt_path = final_ckpt
        else:
            if not include_unsolved:
                print(f"  Skipping {run_name} (unsolved and include_unsolved=False)")
                continue
            ckpt_path = max(ckpts, key=os.path.getctime)

        print(f"  Using checkpoint: {ckpt_path} (solved={solved})")

        # Setup Env
        base_env = HallOfMirrorsGridworld(
            random_rot=False, random_step=False, random_val=False
        )
        env = ManualFrameStack(base_env)
        
        # Increase Total Steps to 6M (2M + 2M + 2M)
        # Note: 'total_steps' in config is the *limit*. 
        # The agent.train() loop continues from agent.total_env_steps.
        
        cfg = PPOConfig(total_steps=HALL_STEPS) # Placeholder, updated per phase
        agent = PPOAgent(env, hidden_size=hs, n_hidden_layers=nl, config=cfg)
        agent.load_checkpoint(ckpt_path)

        adapt_ckpt_dir = os.path.join(MIRRORS_CHECKPOINT_DIR, run_name)
        os.makedirs(adapt_ckpt_dir, exist_ok=True)

        all_stats = []

        def make_plot_cb():
            def _plot_cb(stats):
                if not stats: return
                df = pd.DataFrame(stats)
                # Append current stats to global tracking if needed, or just save csv
                csv_path = os.path.join(MIRRORS_DATA_DIR, f"{run_name}_hall_progress.csv")
                
                # If file exists, append? No, 'stats' grows accumulatively in agent.train
                # actually agent.train returns the *new* stats from that call.
                # To keep it simple, we just overwrite the CSV with the cumulative stats 
                # from the current train call, but that misses previous stages.
                # Better: let's rely on agent.total_env_steps to be consistent.
                
                # We will just append new data to the CSV manually after each stage.
                pass
            return _plot_cb

        # 1. Pre Evaluation
        pre_return = evaluate_agent(agent, env, n_episodes=30)
        print(f"  [Pre] avg return: {pre_return:.2f}")

        # 2. Grace Period
        print(f"  Running grace phase for {GRACE_STEPS} steps...")
        agent.config.total_steps = agent.total_env_steps + GRACE_STEPS
        stats_grace = agent.train(
            verbose=False, stop_at_return=None, checkpoint_dir=adapt_ckpt_dir
        )
        # Save Grace CSV
        pd.DataFrame(stats_grace).to_csv(
            os.path.join(MIRRORS_DATA_DIR, f"{run_name}_grace_progress.csv"), index=False
        )
        grace_return = evaluate_agent(agent, env, n_episodes=30)
        print(f"  [After grace] avg return: {grace_return:.2f}")

        # ----------------------------------------------------
        # STAGE 1: Random Rotation (Step & Val Fixed)
        # ----------------------------------------------------
        print(f"  [Stage 1] Random Rotation ({STAGE_BUDGET} Steps)...")
        base_env.random_rot = True
        base_env.random_step = False
        base_env.random_val = False
        
        agent.config.total_steps = agent.total_env_steps + STAGE_BUDGET
        stats_s1 = agent.train(verbose=True, checkpoint_dir=adapt_ckpt_dir)
        
        # ----------------------------------------------------
        # STAGE 2: Random Step Size (Rot & Val Fixed)
        # ----------------------------------------------------
        print(f"  [Stage 2] Random Step Size ({STAGE_BUDGET} Steps)...")
        base_env.random_rot = False
        base_env.random_step = True
        base_env.random_val = False
        
        agent.config.total_steps = agent.total_env_steps + STAGE_BUDGET
        stats_s2 = agent.train(verbose=True, checkpoint_dir=adapt_ckpt_dir)

        # ----------------------------------------------------
        # STAGE 3: Random Value Map (Rot & Step Fixed)
        # ----------------------------------------------------
        print(f"  [Stage 3] Random Value Map ({STAGE_BUDGET} Steps)...")
        base_env.random_rot = False
        base_env.random_step = False
        base_env.random_val = True
        
        agent.config.total_steps = agent.total_env_steps + STAGE_BUDGET
        stats_s3 = agent.train(verbose=True, checkpoint_dir=adapt_ckpt_dir)

        # Combine Hall Stats
        full_hall_stats = stats_s1 + stats_s2 + stats_s3
        pd.DataFrame(full_hall_stats).to_csv(
            os.path.join(MIRRORS_DATA_DIR, f"{run_name}_hall_progress.csv"), index=False
        )
        
        # Final Eval
        hall_return = evaluate_agent(agent, env, n_episodes=50)
        print(f"  [Post-hall] avg return: {hall_return:.2f}")

        # Save result for THIS run
        result_data = [{
            "hidden_size": hs,
            "layers": nl,
            "solved_pretrain": solved,
            "pre_return": pre_return,
            "grace_return": grace_return,
            "hall_return": hall_return,
        }]
        
        outfile = os.path.join(MIRRORS_DATA_DIR, f"mirrors_summary_{run_name}.csv")
        pd.DataFrame(result_data).to_csv(outfile, index=False)
        print(f"Saved mirrors summary to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-unsolved", action="store_true")
    args = parser.parse_args()
    run_batch_mirrors(include_unsolved=args.include_unsolved)