# ============================================
# File: utils/plot_story.py
# ============================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap

from shared.config import PRETRAIN_DATA_DIR, MIRRORS_DATA_DIR, PRETRAIN_TARGET_RETURN

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Formatting Helpers ---
GAUGE_LABELS = {
    "rotation": "Rotation",
    "step_size": "Step Size",
    "reward_map": "Reward Map",
    "nuisance": "Nuisance (Noise)",
    "dist_to_wall": "Dist. to Wall"
}

def format_gauge_name(name):
    return GAUGE_LABELS.get(name, name.replace("_", " ").title())

def calc_score(row):
    s = row.get('sensitivity', 0)
    m = row.get('morphism', 0)
    d = row.get('decodability', 0)
    d = max(0.0, min(1.0, d))
    m = max(0.0, m)
    return s * m * (1.0 - d)

def load_all_data():
    # 1. Load Pretrain Gauges
    pre_files = glob.glob(os.path.join(PRETRAIN_DATA_DIR, "gauge_analysis_hs*_l*.csv"))
    df_pre_list = []
    for f in pre_files:
        df = pd.read_csv(f)
        df['phase'] = 'Pretrain'
        df['gauge_score'] = df.apply(calc_score, axis=1)
        df_pre_list.append(df)
    df_pre = pd.concat(df_pre_list, ignore_index=True) if df_pre_list else pd.DataFrame()
    
    if 'num_layers' in df_pre.columns:
        df_pre.rename(columns={'num_layers': 'layers'}, inplace=True)

    # 2. Load Stage Gauges
    stage_files = glob.glob(os.path.join(MIRRORS_DATA_DIR, "gauge_analysis_stages_*.csv"))
    df_stage_list = []
    for f in stage_files:
        df = pd.read_csv(f)
        df['phase'] = df['stage']
        df['gauge_score'] = df.apply(calc_score, axis=1)
        df_stage_list.append(df)
    df_stage = pd.concat(df_stage_list, ignore_index=True) if df_stage_list else pd.DataFrame()

    if 'num_layers' in df_stage.columns:
        df_stage.rename(columns={'num_layers': 'layers'}, inplace=True)

    # 3. Load Summaries (Performance)
    sum_files = glob.glob(os.path.join(MIRRORS_DATA_DIR, "mirrors_summary_*.csv"))
    df_sum_list = []
    
    for f in sum_files:
        summ = pd.read_csv(f)
        
        if 'rec_rot' not in summ.columns:
            run_name = os.path.basename(f).replace("mirrors_summary_", "").replace(".csv", "")
            hall_csv = os.path.join(MIRRORS_DATA_DIR, f"{run_name}_hall_progress.csv")
            
            # Defaults
            rec_rot, rec_step, rec_val = 0.0, 0.0, 0.0
            
            if os.path.exists(hall_csv):
                try:
                    hall_df = pd.read_csv(hall_csv)
                    if not hall_df.empty:
                        total = len(hall_df)
                        chunk = total // 3
                        
                        rec_rot = hall_df.iloc[:chunk]['rolling_return'].max()
                        rec_step = hall_df.iloc[chunk:2*chunk]['rolling_return'].max()
                        rec_val = hall_df.iloc[2*chunk:]['rolling_return'].max()
                except:
                    pass 
            
            summ['rec_rot'] = rec_rot
            summ['rec_step'] = rec_step
            summ['rec_val'] = rec_val
            
        df_sum_list.append(summ)
        
    df_perf = pd.concat(df_sum_list, ignore_index=True) if df_sum_list else pd.DataFrame()

    for col in ['rec_rot', 'rec_step', 'rec_val']:
        if col not in df_perf.columns:
            df_perf[col] = 0.0

    # Apply Clean Labels globally
    if not df_pre.empty:
        df_pre['gauge_type'] = df_pre['gauge_type'].apply(format_gauge_name)
    if not df_stage.empty:
        df_stage['gauge_type'] = df_stage['gauge_type'].apply(format_gauge_name)

    return df_pre, df_stage, df_perf

def plot_fig2_baseline(df_pre):
    plt.figure(figsize=(11, 7))
    order = [GAUGE_LABELS["dist_to_wall"], GAUGE_LABELS["nuisance"], 
             GAUGE_LABELS["rotation"], GAUGE_LABELS["step_size"], GAUGE_LABELS["reward_map"]]
    
    sns.boxplot(data=df_pre, x="gauge_type", y="gauge_score", order=order, color="white", showfliers=False, zorder=1)
    
    # Custom Colormap: Orange -> Red -> Purple
    cmap_custom = LinearSegmentedColormap.from_list("orange_red_purple", ["orange", "red", "purple"])
    
    # Map hidden_size to colors using linear scale
    norm = Normalize(vmin=df_pre['hidden_size'].min(), vmax=df_pre['hidden_size'].max())
    
    # Create scatter plot with jitter
    for i, gauge in enumerate(order):
        data = df_pre[df_pre['gauge_type'] == gauge]
        x = np.random.normal(i, 0.04, size=len(data))  # jitter
        scatter = plt.scatter(x, data['gauge_score'], 
                            c=data['hidden_size'], 
                            cmap=cmap_custom, 
                            norm=norm,
                            s=80, 
                            alpha=0.8,
                            edgecolors='none',
                            zorder=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=plt.gca())
    cbar.set_label('Hidden Size', fontsize=12)
    
    plt.title("Figure 2: Baseline Gauge Identification (Pre-Adaptation)", fontsize=14)
    plt.ylabel("Gauge Score", fontsize=12)
    plt.xlabel("Feature Type", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_baseline_identification.png"), dpi=300)
    plt.close()

def plot_fig3_performance(hs, nl):
    run_name = f"hs{hs}_l{nl}"
    
    pre_path = os.path.join(PRETRAIN_DATA_DIR, f"{run_name}_progress.csv")
    grace_path = os.path.join(MIRRORS_DATA_DIR, f"{run_name}_grace_progress.csv")
    hall_path = os.path.join(MIRRORS_DATA_DIR, f"{run_name}_hall_progress.csv")
    
    if not (os.path.exists(pre_path) and os.path.exists(grace_path) and os.path.exists(hall_path)):
        print(f"Skipping Fig 2 (Missing CSVs for {run_name})")
        return

    df_pre = pd.read_csv(pre_path)
    df_grace = pd.read_csv(grace_path)
    df_hall = pd.read_csv(hall_path)
    
    df_full = pd.concat([df_pre, df_grace, df_hall], ignore_index=True)
    
    plt.figure(figsize=(12, 5))
    plt.plot(df_full['step'], df_full['rolling_return'], color='black', linewidth=1.5, zorder=10)
    
    # Calculate span boundaries
    pre_min, pre_max = df_pre['step'].min(), df_pre['step'].max()
    grace_min, grace_max = df_grace['step'].min(), df_grace['step'].max()
    hall_min, hall_max = df_hall['step'].min(), df_hall['step'].max()

    # Colors: Purple -> Orange -> Red
    plt.axvspan(pre_min, pre_max, color='purple', alpha=0.1, label='Pretrain')
    plt.axvspan(grace_min, grace_max, color='orange', alpha=0.2, label='Grace')
    plt.axvspan(hall_min, hall_max, color='red', alpha=0.1, label='Hall of Mirrors')
    
    plt.title(f"Figure 3: Adaptation Profile (Agent: Hidden {hs}, Layers {nl})", fontsize=14)
    plt.xlabel("Total Environment Steps", fontsize=12)
    plt.ylabel("Rolling Return", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig3_performance_hs{hs}_l{nl}.png"), dpi=300)
    plt.close()

def plot_fig4_grid(df_pre, df_stage, df_perf):
    if df_perf.empty: return

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    features = [
        ("rotation", "stage_1_rot", "rec_rot", "Rotation"),
        ("step_size", "stage_2_step", "rec_step", "Step Size"),
        ("reward_map", "stage_3_val", "rec_val", "Reward Map")
    ]
    
    # Custom Palette for Layers: 1=Orange, 2=Red, 3=Purple
    layer_palette = {1: "orange", 2: "red", 3: "purple"}

    for i, (raw_gauge, stage_name, rec_col, clean_name) in enumerate(features):
        # --- LEFT: Performance ---
        df_perf['recovery_pct'] = df_perf[rec_col] / PRETRAIN_TARGET_RETURN
        
        ax_perf = axes[i, 0]
        sns.lineplot(data=df_perf, x="hidden_size", y="recovery_pct", hue="layers", 
                    palette=layer_palette, marker="o", ax=ax_perf)
        ax_perf.set_title(f"Max Performance Recovery ({clean_name})", fontsize=12)
        ax_perf.set_ylim(0, 1.2)
        ax_perf.set_ylabel("% of Baseline")
        ax_perf.set_xlabel("Hidden Size")
        ax_perf.legend(title="Layers")
        ax_perf.grid(True, alpha=0.3)
        
        # --- RIGHT: Reification ---
        if df_stage.empty or df_pre.empty:
            continue

        d_pre = df_pre[df_pre['gauge_type'] == clean_name][['hidden_size', 'layers', 'gauge_score']]
        d_post = df_stage[
            (df_stage['gauge_type'] == clean_name) & 
            (df_stage['stage'] == stage_name)
        ][['hidden_size', 'layers', 'gauge_score']]
        
        if not d_pre.empty and not d_post.empty:
            merged = pd.merge(d_pre, d_post, on=['hidden_size', 'layers'], suffixes=('_pre', '_post'))
            merged['reification'] = (merged['gauge_score_pre'] - merged['gauge_score_post'])
            
            ax_reif = axes[i, 1]
            sns.lineplot(data=merged, x="hidden_size", y="reification", hue="layers",
                        palette=layer_palette, marker="o", ax=ax_reif)
            ax_reif.set_title(f"Gauge Reification ({clean_name})", fontsize=12)
            ax_reif.set_ylabel("Reification Score")
            ax_reif.set_xlabel("Hidden Size")
            ax_reif.legend(title="Layers")
            ax_reif.axhline(0, color='black', linewidth=0.5, linestyle='--')
            ax_reif.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_grid.png"), dpi=300)
    plt.close()

def generate_table_data(df_pre, df_stage):
    if df_pre.empty: return
    summ_pre = df_pre.groupby('gauge_type')['gauge_score'].agg(['mean', 'std'])
    summ_pre.columns = ['Pre_Mean', 'Pre_Std']
    
    if not df_stage.empty:
        s1 = df_stage[df_stage['stage'] == 'stage_1_rot']
        summ_s1 = s1.groupby('gauge_type')['gauge_score'].agg(['mean', 'std'])
        summ_s1.columns = ['S1_Mean', 'S1_Std']
        
        s2 = df_stage[df_stage['stage'] == 'stage_2_step']
        summ_s2 = s2.groupby('gauge_type')['gauge_score'].agg(['mean', 'std'])
        summ_s2.columns = ['S2_Mean', 'S2_Std']
        
        s3 = df_stage[df_stage['stage'] == 'stage_3_val']
        summ_s3 = s3.groupby('gauge_type')['gauge_score'].agg(['mean', 'std'])
        summ_s3.columns = ['S3_Mean', 'S3_Std']
        
        full_table = pd.concat([summ_pre, summ_s1, summ_s2, summ_s3], axis=1)
    else:
        full_table = summ_pre

    csv_path = os.path.join(OUTPUT_DIR, "table1_gauge_evolution.csv")
    full_table.to_csv(csv_path)
    print(f"Table data saved to {csv_path}")

def main():
    print("Loading Data...")
    df_pre, df_stage, df_perf = load_all_data()
    
    print("Generating Figure 2...")
    if not df_pre.empty:
        plot_fig2_baseline(df_pre)
    
    print("Generating Figure 3...")
    target_hs, target_nl = 64, 2
    plot_fig3_performance(target_hs, target_nl)
    
    print("Generating Figure 4...")
    plot_fig4_grid(df_pre, df_stage, df_perf)
    
    print("Generating Table Data...")
    generate_table_data(df_pre, df_stage)
    
    print("Done.")

if __name__ == "__main__":
    main()