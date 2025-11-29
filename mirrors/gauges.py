# ============================================
# File: mirrors/gauges.py
# ============================================

import glob
import os
import torch
import pandas as pd
from shared.config import (
    COMPLEXITIES, GRACE_STEPS,
    PRETRAIN_CHECKPOINT_DIR, MIRRORS_CHECKPOINT_DIR, MIRRORS_DATA_DIR
)
from shared.env import HallOfMirrorsGridworld, ManualFrameStack
from shared.agent import PPOAgent
from pretrain.gauges import collect_gauge_dataset, compute_metrics

STAGE_DURATION = 2_000_000

def get_step_from_filename(filepath):
    """Extracts 12345 from '.../ckpt_step_12345.pt'"""
    base = os.path.basename(filepath)
    if not base.startswith("ckpt_step_"): return -1
    try:
        return int(base.replace("ckpt_step_", "").replace(".pt", ""))
    except:
        return -1

def find_stage_checkpoints(run_name, hs, nl):
    """
    Calculates the step boundaries for this specific run and finds
    the closest existing checkpoint for each stage end.
    """
    # 1. Determine T_0 (End of Pretraining)
    pre_dir = os.path.join(PRETRAIN_CHECKPOINT_DIR, run_name)
    final_ckpt = os.path.join(pre_dir, "final_solved.pt")
    
    start_steps = 0
    if os.path.exists(final_ckpt):
        # Load lightweight to get steps
        state = torch.load(final_ckpt, map_location='cpu', weights_only=False)
        start_steps = state.get("total_env_steps", 0)
    else:
        # Fallback: find latest numbered ckpt in pretrain
        all_pre = glob.glob(os.path.join(pre_dir, "ckpt_step_*.pt"))
        if not all_pre:
            print(f"  Warning: No pretrain start found for {run_name}")
            return {}
        latest = max(all_pre, key=get_step_from_filename)
        start_steps = get_step_from_filename(latest)

    # 2. Define Targets
    # T_0 + Grace + Stage 1
    target_s1 = start_steps + GRACE_STEPS + STAGE_DURATION
    # Target 1 + Stage 2
    target_s2 = target_s1 + STAGE_DURATION
    # Target 2 + Stage 3
    target_s3 = target_s2 + STAGE_DURATION

    targets = {
        "stage_1_rot": target_s1,
        "stage_2_step": target_s2,
        "stage_3_val": target_s3
    }

    # 3. Find Closest Mirrors Checkpoints
    mir_dir = os.path.join(MIRRORS_CHECKPOINT_DIR, run_name)
    all_mirrors = glob.glob(os.path.join(mir_dir, "ckpt_step_*.pt"))
    
    if not all_mirrors:
        return {}

    # Map {step: path}
    step_map = {get_step_from_filename(p): p for p in all_mirrors if get_step_from_filename(p) > 0}
    if not step_map: return {}
    
    available_steps = list(step_map.keys())
    
    found_checkpoints = {}
    
    for stage_name, target_step in targets.items():
        # Find closest step
        closest_step = min(available_steps, key=lambda x: abs(x - target_step))
        distance = abs(closest_step - target_step)
        
        # Sanity Check: If the closest checkpoint is > 1M steps away, 
        # the run probably crashed or hasn't reached that stage yet.
        if distance > 1_000_000:
            print(f"  Skipping {stage_name}: Closest ckpt is {distance} steps away (incomplete run?)")
            continue
            
        found_checkpoints[stage_name] = step_map[closest_step]
        print(f"  Map {stage_name}: Target {target_step} -> Found {closest_step} (Diff: {distance})")

    return found_checkpoints


def run_post_mirrors_analysis(target_configs=None):
    if target_configs is None:
        target_configs = COMPLEXITIES

    print(f"Searching for checkpoints in {MIRRORS_CHECKPOINT_DIR}...")
    gauge_types = ["rotation", "step_size", "reward_map", "nuisance", "dist_to_wall"]

    for (hs, nl) in target_configs:
        run_name = f"hs{hs}_l{nl}"
        
        # Use the math logic to find files
        stage_map = find_stage_checkpoints(run_name, hs, nl)
        
        if not stage_map:
            print(f"Skipping {run_name} (no valid stage checkpoints found)")
            continue

        print(f"\n=== Analyzing {run_name} Stages ===")
        
        stage_results = []

        for stage_name, ckpt_path in stage_map.items():
            print(f"  > Analyzing {stage_name}...")
            
            # Load Agent
            # Note: We use Fixed env for analysis regardless of training stage
            base_env = HallOfMirrorsGridworld(random_rot=False, random_step=False, random_val=False)
            env = ManualFrameStack(base_env)
            agent = PPOAgent(env, hidden_size=hs, n_hidden_layers=nl)
            agent.load_checkpoint(ckpt_path)

            # Collect Data
            records = collect_gauge_dataset(agent)

            for g_type in gauge_types:
                sens, dec, morph = compute_metrics(records, g_type)
                stage_results.append({
                    "hidden_size": hs,
                    "num_layers": nl,
                    "stage": stage_name,
                    "gauge_type": g_type,
                    "sensitivity": sens,
                    "decodability": dec,
                    "morphism": morph,
                })

        # Save Result
        if stage_results:
            outfile = os.path.join(MIRRORS_DATA_DIR, f"gauge_analysis_stages_{run_name}.csv")
            pd.DataFrame(stage_results).to_csv(outfile, index=False)
            print(f"Saved stage analysis to {outfile}")

if __name__ == "__main__":
    run_post_mirrors_analysis()