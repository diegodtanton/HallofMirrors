# ============================================
# File: pretrain/gauges.py
# ============================================

import glob
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from shared.config import (
    COMPLEXITIES,
    PRETRAIN_CHECKPOINT_DIR,
    PRETRAIN_DATA_DIR,
    GRID_SIZE, MAX_STEPS_PER_EPISODE, 
    N_GOOD_TILES, N_BAD_TILES, FRAME_STACK_SIZE
)
from shared.env import HallOfMirrorsGridworld, ManualFrameStack
from shared.agent import PPOAgent

def collect_gauge_dataset(
    agent: PPOAgent, n_episodes_per_setting: int = 5, max_steps: int = 50
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    rng = np.random.RandomState(42)

    base_env = HallOfMirrorsGridworld(
        grid_size=GRID_SIZE,
        max_steps=MAX_STEPS_PER_EPISODE,
        n_good_tiles=N_GOOD_TILES,
        n_bad_tiles=N_BAD_TILES,
        # Fixed defaults
        random_rot=False, random_step=False, random_val=False,
        fixed_sensor_rotation=0, fixed_step_size=1, fixed_good_is_red=False,
    )
    env = ManualFrameStack(base_env, num_stack=FRAME_STACK_SIZE)

    # --- 1. Gauge Variations ---
    for layout_idx in range(n_episodes_per_setting):
        base_seed = rng.randint(0, 100000)
        
        # Pre-calculate actions
        actions = [rng.randint(0, env.action_space.n) for _ in range(max_steps)]

        # A. Rotation
        for rot in [0, 1, 2, 3]:
            base_env.reseed(base_seed)
            base_env.fixed_sensor_rotation = rot
            base_env.fixed_step_size = 1
            base_env.fixed_good_is_red = False
            
            obs, _ = env.reset()
            obs = obs.astype(np.float32)
            prev_r = 0.0
            
            for t, act in enumerate(actions):
                # Pass prev_reward to latent
                z = agent.get_latent(obs[None, ...], prev_reward_val=prev_r)[0]
                records.append({
                    "z": z, "val": rot, "layout": layout_idx, "step_id": t, "type": "rotation",
                })
                obs, reward, term, trunc, _ = env.step(act)
                obs = obs.astype(np.float32)
                prev_r = float(reward)
                if term or trunc: break

        # B. Step Size
        for step in [1, 2]:
            base_env.reseed(base_seed)
            base_env.fixed_sensor_rotation = 0
            base_env.fixed_step_size = step
            base_env.fixed_good_is_red = False
            
            obs, _ = env.reset()
            obs = obs.astype(np.float32)
            prev_r = 0.0
            
            for t, act in enumerate(actions):
                z = agent.get_latent(obs[None, ...], prev_reward_val=prev_r)[0]
                records.append({
                    "z": z, "val": step, "layout": layout_idx, "step_id": t, "type": "step_size",
                })
                obs, reward, term, trunc, _ = env.step(act)
                obs = obs.astype(np.float32)
                prev_r = float(reward)
                if term or trunc: break

        # C. Reward Map
        for is_red in [True, False]:
            base_env.reseed(base_seed)
            base_env.fixed_sensor_rotation = 0
            base_env.fixed_step_size = 1
            base_env.fixed_good_is_red = is_red
            
            obs, _ = env.reset()
            obs = obs.astype(np.float32)
            prev_r = 0.0
            
            val_int = 1 if is_red else 0
            for t, act in enumerate(actions):
                z = agent.get_latent(obs[None, ...], prev_reward_val=prev_r)[0]
                records.append({
                    "z": z, "val": val_int, "layout": layout_idx, "step_id": t, "type": "reward_map",
                })
                obs, reward, term, trunc, _ = env.step(act)
                obs = obs.astype(np.float32)
                prev_r = float(reward)
                if term or trunc: break

    # --- 2. Nuisance (Noise Channel) & Explicit Feature ---
    # We iterate over layouts first (Geometry)
    for layout_idx in range(n_episodes_per_setting):
        base_seed = rng.randint(0, 100000)
        
        # 1. Establish Geometry & Actions
        # We need to peek at the geometry to get actions, so we reset once.
        base_env.reseed(base_seed)
        base_env.reset()
        actions = [rng.randint(0, env.action_space.n) for _ in range(max_steps)]
        
        # 2. Cycle through NOISE patterns on this exact geometry
        # This keeps 'dist_to_wall' constant per step_id, but changes pixels.
        for noise_seed in [100, 200, 300, 400]:
            base_env.reseed(base_seed)
            base_env.fixed_sensor_rotation = 0
            base_env.fixed_step_size = 1
            base_env.fixed_good_is_red = False
            
            # Reset generates the first observation with default noise
            base_env.reset()
            
            # Force the specific noise pattern we want
            base_env.force_noise_pattern(noise_seed)
            
            # Refresh the frame stack manually to clear old noise
            fresh_obs = base_env._get_obs()
            env.frames.clear()
            for _ in range(FRAME_STACK_SIZE):
                env.frames.append(fresh_obs)
            obs = env._get_ob().astype(np.float32)
            
            prev_r = 0.0
            
            for t, act in enumerate(actions):
                z = agent.get_latent(obs[None, ...], prev_reward_val=prev_r)[0]
                
                # --- explicit feature calculation ---
                y, x = base_env.agent_pos
                def ray(dy, dx):
                    d = 0
                    cy, cx = y, x
                    while True:
                        cy += dy; cx += dx
                        if not (0 <= cy < base_env.grid_size and 0 <= cx < base_env.grid_size): break 
                        if base_env.grid[cy, cx] == 1: break
                        d += 1
                    return d
                
                min_dist = min(ray(-1, 0), ray(1, 0), ray(0, -1), ray(0, 1))
                
                # Record Noise Nuisance (val = noise_seed)
                records.append({
                    "z": z, "val": noise_seed, "layout": layout_idx, "step_id": t, "type": "nuisance",
                })
                
                # Record Explicit Feature (val = distance)
                records.append({
                    "z": z, "val": min_dist, "layout": layout_idx, "step_id": t, "type": "dist_to_wall", 
                })

                obs, reward, term, trunc, _ = env.step(act)
                obs = obs.astype(np.float32)
                prev_r = float(reward)
                if term or trunc: break

    return records


def compute_metrics(records: List[Dict[str, Any]], gauge_type: str):
    data = [r for r in records if r["type"] == gauge_type]
    if not data:
        return 0.0, 0.0, 0.0

    # --- 1. Sensitivity ---
    # For 'dist_to_wall', sensitivity isn't defined via intervention.
    if gauge_type == "dist_to_wall":
        sens = 0.0
    else:
        aligned_groups = {}
        for r in data:
            key = (r["layout"], r["step_id"])
            aligned_groups.setdefault(key, []).append(r["z"])
        
        dists = []
        for zs in aligned_groups.values():
            if len(zs) < 2: continue
            stack = np.stack(zs)
            mean_z = np.mean(stack, axis=0)
            dist = np.mean(np.linalg.norm(stack - mean_z, axis=1))
            dists.append(dist)
        sens = float(np.mean(dists)) if dists else 0.0

    # --- 2. Decodability ---
    X = np.stack([r["z"] for r in data])
    y_raw = np.array([r["val"] for r in data])
    unique_y, y_int = np.unique(y_raw, return_inverse=True)
    n_classes = len(unique_y)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    
    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y_int[train_idx]).long()
    X_test = torch.from_numpy(X[test_idx]).float()
    y_test = torch.from_numpy(y_int[test_idx]).long()

    if n_classes < 2:
        dec = 0.0
    else:
        probe = nn.Linear(X.shape[1], n_classes)
        opt = optim.Adam(probe.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        
        for _ in range(200):
            logits = probe(X_train)
            loss = loss_fn(logits, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        with torch.no_grad():
            preds = probe(X_test).argmax(dim=-1)
            dec = float((preds == y_test).float().mean().item())

    # --- 3. Morphism ---
    # Only calculate morphism for true gauges, not explicit/nuisance
    if gauge_type in ["dist_to_wall", "nuisance"]:
        return sens, dec, 0.0

    vals = sorted(list(set(y_raw)))
    if len(vals) < 2: 
        return sens, dec, 0.0
    
    v0, v1 = vals[0], vals[1]
    
    exact_map = {}
    for r in data:
        k = (r["layout"], r["step_id"])
        exact_map.setdefault(k, {})[r["val"]] = r["z"]
        
    Z0, Z1 = [], []
    for k, val_dict in exact_map.items():
        if v0 in val_dict and v1 in val_dict:
            Z0.append(val_dict[v0])
            Z1.append(val_dict[v1])

    if len(Z0) < 10: 
        return sens, dec, 0.0
        
    Z0 = np.stack(Z0)
    Z1 = np.stack(Z1)
    
    idx = np.arange(len(Z0))
    np.random.shuffle(idx)
    split = int(0.8 * len(Z0))
    train_idx, test_idx = idx[:split], idx[split:]

    mapper = nn.Linear(Z0.shape[1], Z1.shape[1])
    opt = optim.Adam(mapper.parameters(), lr=1e-2)
    for _ in range(300):
        pred = mapper(torch.from_numpy(Z0[train_idx]).float())
        loss = ((pred - torch.from_numpy(Z1[train_idx]).float()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = mapper(torch.from_numpy(Z0[test_idx]).float()).numpy()
    
    target = Z1[test_idx]
    ss_res = float(((target - preds) ** 2).sum())
    ss_tot = float(((target - target.mean(axis=0)) ** 2).sum() + 1e-8)
    r2 = 1.0 - ss_res / ss_tot
    morph = float(max(r2, -1.0))
    
    return sens, dec, morph


def run_gauge_analysis(target_configs=None):
    if target_configs is None:
        target_configs = COMPLEXITIES

    print(f"Searching for checkpoints in {PRETRAIN_CHECKPOINT_DIR}...")
    gauge_types = ["rotation", "step_size", "reward_map", "nuisance", "dist_to_wall"]

    for (hs, nl) in target_configs:
        run_name = f"hs{hs}_l{nl}"
        run_dir = os.path.join(PRETRAIN_CHECKPOINT_DIR, run_name)

        ckpts = glob.glob(os.path.join(run_dir, "*.pt"))
        if not ckpts:
            print(f"Skipping {run_name} (no checkpoints)")
            continue

        final_ckpt = os.path.join(run_dir, "final_solved.pt")
        if os.path.exists(final_ckpt):
            ckpt_path = final_ckpt
        else:
            ckpt_path = max(ckpts, key=os.path.getctime)

        print(f"\nAnalyzing {run_name} from {ckpt_path} ...")

        base_env = HallOfMirrorsGridworld(random_rot=False, random_step=False, random_val=False)
        env = ManualFrameStack(base_env)
        agent = PPOAgent(env, hidden_size=hs, n_hidden_layers=nl)
        agent.load_checkpoint(ckpt_path)

        records = collect_gauge_dataset(agent)
        run_results = []

        for g_type in gauge_types:
            sens, dec, morph = compute_metrics(records, g_type)
            print(f"  > {g_type:12s} | S: {sens:.3f} | D: {dec:.3f} | M: {morph:.3f}")

            run_results.append({
                "hidden_size": hs,
                "num_layers": nl,
                "gauge_type": g_type,
                "sensitivity": sens,
                "decodability": dec,
                "morphism": morph,
            })

        if run_results:
            outfile = os.path.join(PRETRAIN_DATA_DIR, f"gauge_analysis_{run_name}.csv")
            df = pd.DataFrame(run_results)
            df.to_csv(outfile, index=False)
            print(f"Saved analysis to {outfile}")

if __name__ == "__main__":
    run_gauge_analysis()