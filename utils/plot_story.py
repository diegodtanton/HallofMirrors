# ============================================
# File: utils/plot_story.py
# ============================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.colors import Normalize, LinearSegmentedColormap

from shared.config import PRETRAIN_TARGET_RETURN

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the three experimental runs
DATA_FOLDERS = [
    "2025-11-28_23-01-05", # Run 1
    "2025-12-08_22-56-02", # Run 2
    "2025-12-09_09-59-37"  # Run 3
]

# --- Formatting Helpers ---
GAUGE_LABELS = {
    "rotation": "Rotation",
    "step_size": "Step Size",
    "reward_map": "Reward Map",
    "nuisance": "Nuisance (Noise)",
    "dist_to_wall": "Dist. to Wall"
}

LAYER_PALETTE = {1: "orange", 2: "red", 3: "purple"}

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
    all_pre = []
    all_stage = []
    all_perf = []

    for run_idx, folder_name in enumerate(DATA_FOLDERS):
        run_id = f"run_{run_idx+1}"
        base_path = os.path.join("downloaded_results", folder_name)
        
        pre_dir = os.path.join(base_path, "pretrain", "data")
        mir_dir = os.path.join(base_path, "mirrors", "data")

        # 1. Load Pretrain Gauges
        pre_files = glob.glob(os.path.join(pre_dir, "gauge_analysis_hs*_l*.csv"))
        for f in pre_files:
            df = pd.read_csv(f)
            df['phase'] = 'Pretrain'
            df['run_id'] = run_id
            df['gauge_score'] = df.apply(calc_score, axis=1)
            all_pre.append(df)

        # 2. Load Stage Gauges
        stage_files = glob.glob(os.path.join(mir_dir, "gauge_analysis_stages_*.csv"))
        for f in stage_files:
            df = pd.read_csv(f)
            df['phase'] = df['stage']
            df['run_id'] = run_id
            df['gauge_score'] = df.apply(calc_score, axis=1)
            all_stage.append(df)

        # 3. Load Summaries (Performance)
        sum_files = glob.glob(os.path.join(mir_dir, "mirrors_summary_*.csv"))
        for f in sum_files:
            summ = pd.read_csv(f)
            summ['run_id'] = run_id
            
            # Helper to extract max recovery if summary doesn't have it calculated yet
            if 'rec_rot' not in summ.columns:
                run_name = os.path.basename(f).replace("mirrors_summary_", "").replace(".csv", "")
                hall_csv = os.path.join(mir_dir, f"{run_name}_hall_progress.csv")
                
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
            
            all_perf.append(summ)

    df_pre = pd.concat(all_pre, ignore_index=True) if all_pre else pd.DataFrame()
    df_stage = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    df_perf = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()

    if 'num_layers' in df_pre.columns: df_pre.rename(columns={'num_layers': 'layers'}, inplace=True)
    if 'num_layers' in df_stage.columns: df_stage.rename(columns={'num_layers': 'layers'}, inplace=True)

    for col in ['rec_rot', 'rec_step', 'rec_val']:
        if col not in df_perf.columns: df_perf[col] = 0.0

    if not df_pre.empty: df_pre['gauge_type'] = df_pre['gauge_type'].apply(format_gauge_name)
    if not df_stage.empty: df_stage['gauge_type'] = df_stage['gauge_type'].apply(format_gauge_name)

    return df_pre, df_stage, df_perf

def calculate_significance(df, x_col, y_col, group_col):
    auc_data = []
    run_ids = df['run_id'].unique()
    layers = df[group_col].unique()
    layer_aucs = {l: [] for l in layers}

    for rid in run_ids:
        for l in layers:
            subset = df[(df['run_id'] == rid) & (df[group_col] == l)].sort_values(by=x_col)
            if len(subset) < 2: continue
            
            x = subset[x_col].values
            y = subset[y_col].values
            auc = np.trapz(y, x)
            layer_aucs[l].append(auc)

    valid_lists = [v for k, v in layer_aucs.items() if len(v) >= 2]
    if len(valid_lists) < 2: return "N/A"
        
    f_stat, p_val = stats.f_oneway(*valid_lists)
    
    if p_val < 0.001: return "p < 0.001"
    if p_val < 0.01: return f"p = {p_val:.3f}"
    if p_val < 0.05: return f"p = {p_val:.3f}"
    return "ns"

# ========================================================
# Figure 2: Baseline (Main - Colored by Layer)
# ========================================================
def plot_fig2_main(df_pre):
    plt.figure(figsize=(11, 8))
    order = [GAUGE_LABELS["dist_to_wall"], GAUGE_LABELS["nuisance"], 
             GAUGE_LABELS["rotation"], GAUGE_LABELS["step_size"], GAUGE_LABELS["reward_map"]]
    
    # 1. Underlying Boxplot
    sns.boxplot(data=df_pre, x="gauge_type", y="gauge_score", order=order, color="white", showfliers=False, zorder=1)
    
    # 2. Stripplot colored by LAYERS
    sns.stripplot(data=df_pre, x="gauge_type", y="gauge_score", hue="layers", 
                  order=order, palette=LAYER_PALETTE,
                  jitter=0.2, alpha=0.7, size=6, dodge=False, zorder=2)
    
    # Calculate Max Y for Text
    global_max_y = df_pre['gauge_score'].max()
    text_y_pos = global_max_y * 1.05 

    # 3. Stats (Correlation with LAYERS)
    for i, gauge in enumerate(order):
        data = df_pre[df_pre['gauge_type'] == gauge]
        if len(data) > 2:
            corr, p_val = stats.spearmanr(data['layers'], data['gauge_score'])
            
            if p_val < 0.001: p_str = "p < 0.001"
            else: p_str = f"p = {p_val:.3f}"
            
            label = f"$r_s$ = {corr:.2f}\n{p_str}"
            plt.text(i, text_y_pos, label, 
                     ha='center', va='bottom', fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    plt.ylim(top=text_y_pos * 1.15)
    plt.title(f"Figure 2: Baseline Gauge Identification (Correlation with Depth)", fontsize=14)
    plt.ylabel("Gauge Score", fontsize=12)
    plt.xlabel("Feature Type", fontsize=12)
    plt.legend(title="Layers", loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_baseline_layers.png"), dpi=300)
    plt.close()

# ========================================================
# Figure 2: Supplemental (Colored by Hidden Size)
# ========================================================
def plot_fig2_supp(df_pre):
    plt.figure(figsize=(11, 8))
    order = [GAUGE_LABELS["dist_to_wall"], GAUGE_LABELS["nuisance"], 
             GAUGE_LABELS["rotation"], GAUGE_LABELS["step_size"], GAUGE_LABELS["reward_map"]]
    
    sns.boxplot(data=df_pre, x="gauge_type", y="gauge_score", order=order, color="white", showfliers=False, zorder=1)
    
    cmap_custom = LinearSegmentedColormap.from_list("orange_red_purple", ["orange", "red", "purple"])
    norm = Normalize(vmin=df_pre['hidden_size'].min(), vmax=df_pre['hidden_size'].max())
    
    for i, gauge in enumerate(order):
        data = df_pre[df_pre['gauge_type'] == gauge]
        x_jitter = np.random.normal(i, 0.04, size=len(data)) 
        plt.scatter(x_jitter, data['gauge_score'], c=data['hidden_size'], 
                    cmap=cmap_custom, norm=norm, s=60, alpha=0.7, edgecolors='none', zorder=2)
    
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_custom), ax=plt.gca())
    cbar.set_label('Hidden Size', fontsize=12)
    
    plt.title("Supplemental Fig 2: Baseline Gauge Identification (Colored by Hidden Size)", fontsize=14)
    plt.ylabel("Gauge Score", fontsize=12)
    plt.xlabel("Feature Type", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_baseline_supplemental.png"), dpi=300)
    plt.close()

# ========================================================
# Figure 4A: Performance Stack (3 Rows, 1 Column)
# ========================================================
def plot_performance_stack(df_perf):
    if df_perf.empty: return

    # Taller figsize to accommodate 3 vertical plots
    fig, axes = plt.subplots(3, 1, figsize=(8, 14))
    
    features = [
        ("rotation", "stage_1_rot", "rec_rot", "Rotation"),
        ("step_size", "stage_2_step", "rec_step", "Step Size"),
        ("reward_map", "stage_3_val", "rec_val", "Reward Map")
    ]
    
    for i, (raw_gauge, stage_name, rec_col, clean_name) in enumerate(features):
        df_perf['recovery_pct'] = df_perf[rec_col] / PRETRAIN_TARGET_RETURN
        p_val_perf = calculate_significance(df_perf, 'hidden_size', 'recovery_pct', 'layers')

        ax = axes[i]
        sns.lineplot(data=df_perf, x="hidden_size", y="recovery_pct", hue="layers", 
                     palette=LAYER_PALETTE, marker="o", 
                     errorbar='sd', err_style='bars', err_kws={'capsize': 5}, 
                     ax=ax)
        
        ax.set_title(f"Max Performance Recovery ({clean_name}) | {p_val_perf}", fontsize=12)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("% of Baseline")
        ax.set_xlabel("Hidden Size" if i == 2 else "") # Only label bottom axis
        ax.legend(title="Layers", loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_performance_stack.png"), dpi=300)
    plt.close()

# ========================================================
# Figure 4B: Reification Stack (3 Rows, 1 Column)
# ========================================================
def plot_reification_stack(df_pre, df_stage):
    if df_stage.empty or df_pre.empty: return

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    features = [
        ("rotation", "stage_1_rot", "rec_rot", "Rotation"),
        ("step_size", "stage_2_step", "rec_step", "Step Size"),
        ("reward_map", "stage_3_val", "rec_val", "Reward Map")
    ]
    
    for i, (raw_gauge, stage_name, rec_col, clean_name) in enumerate(features):
        d_pre = df_pre[df_pre['gauge_type'] == clean_name][['hidden_size', 'layers', 'run_id', 'gauge_score']]
        d_post = df_stage[
            (df_stage['gauge_type'] == clean_name) & 
            (df_stage['stage'] == stage_name)
        ][['hidden_size', 'layers', 'run_id', 'gauge_score']]
        
        ax = axes[i]
        
        if not d_pre.empty and not d_post.empty:
            merged = pd.merge(d_pre, d_post, on=['hidden_size', 'layers', 'run_id'], suffixes=('_pre', '_post'))
            merged['reification'] = (merged['gauge_score_pre'] - merged['gauge_score_post'])
            
            p_val_reif = calculate_significance(merged, 'hidden_size', 'reification', 'layers')

            sns.lineplot(data=merged, x="hidden_size", y="reification", hue="layers",
                         palette=LAYER_PALETTE, marker="o", 
                         errorbar='sd', err_style='bars', err_kws={'capsize': 5},
                         ax=ax)
            
            ax.set_title(f"Gauge Reification ({clean_name}) | {p_val_reif}", fontsize=12)
            ax.set_ylabel("Reification Score")
            ax.set_xlabel("Hidden Size" if i == 2 else "")
            ax.legend(title="Layers", loc='upper left')
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_reification_stack.png"), dpi=300)
    plt.close()

# ========================================================
# Table Generators
# ========================================================
def generate_table_avg(df_pre, df_stage):
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
    print(f"Main table saved to {csv_path}")

def generate_detailed_component_tables(df_pre, df_stage):
    if df_pre.empty or df_stage.empty: return

    mapping = {
        "Rotation": "stage_1_rot",
        "Step Size": "stage_2_step",
        "Reward Map": "stage_3_val"
    }

    # --- Table 3: By Layer (Depth) ---
    md_lines_l = []
    md_lines_l.append("# Table 3: Component Analysis by Depth (Layers)")
    md_lines_l.append("| Feature | Layers | Pre Sens | Pre Dec | Pre Morph | Post Sens | Post Dec | Post Morph |")
    md_lines_l.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for gauge_name, stage_id in mapping.items():
        pre_sub = df_pre[df_pre['gauge_type'] == gauge_name]
        post_sub = df_stage[(df_stage['gauge_type'] == gauge_name) & (df_stage['stage'] == stage_id)]
        
        if pre_sub.empty or post_sub.empty: continue
        
        # Group by layers
        layers = sorted(pre_sub['layers'].unique())
        for l in layers:
            p_l = pre_sub[pre_sub['layers'] == l]
            po_l = post_sub[post_sub['layers'] == l]
            
            line = (f"| {gauge_name} | {l} | "
                    f"{p_l['sensitivity'].mean():.3f} | {p_l['decodability'].mean():.3f} | {p_l['morphism'].mean():.3f} | "
                    f"{po_l['sensitivity'].mean():.3f} | {po_l['decodability'].mean():.3f} | {po_l['morphism'].mean():.3f} |")
            md_lines_l.append(line)

    with open(os.path.join(OUTPUT_DIR, "table3_components_by_layer.md"), "w") as f:
        f.write("\n".join(md_lines_l))
    print("Table 3 saved.")

    # --- Table 4: By Hidden Size (Width) ---
    md_lines_h = []
    md_lines_h.append("# Table 4: Component Analysis by Width (Hidden Size)")
    md_lines_h.append("| Feature | Hidden | Pre Sens | Pre Dec | Pre Morph | Post Sens | Post Dec | Post Morph |")
    md_lines_h.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for gauge_name, stage_id in mapping.items():
        pre_sub = df_pre[df_pre['gauge_type'] == gauge_name]
        post_sub = df_stage[(df_stage['gauge_type'] == gauge_name) & (df_stage['stage'] == stage_id)]
        
        if pre_sub.empty or post_sub.empty: continue
        
        hiddens = sorted(pre_sub['hidden_size'].unique())
        for h in hiddens:
            p_h = pre_sub[pre_sub['hidden_size'] == h]
            po_h = post_sub[post_sub['hidden_size'] == h]
            
            line = (f"| {gauge_name} | {h} | "
                    f"{p_h['sensitivity'].mean():.3f} | {p_h['decodability'].mean():.3f} | {p_h['morphism'].mean():.3f} | "
                    f"{po_h['sensitivity'].mean():.3f} | {po_h['decodability'].mean():.3f} | {po_h['morphism'].mean():.3f} |")
            md_lines_h.append(line)

    with open(os.path.join(OUTPUT_DIR, "table4_components_by_hidden.md"), "w") as f:
        f.write("\n".join(md_lines_h))
    print("Table 4 saved.")

def main():
    print(f"Loading Data from {len(DATA_FOLDERS)} runs...")
    df_pre, df_stage, df_perf = load_all_data()
    
    print("Generating Figure 2 Main (Layers + Corr)...")
    if not df_pre.empty: plot_fig2_main(df_pre)

    print("Generating Figure 2 Supplemental (Hidden Size)...")
    if not df_pre.empty: plot_fig2_supp(df_pre)
    
    print("Generating Figure 4A (Performance Stack)...")
    plot_performance_stack(df_perf)

    print("Generating Figure 4B (Reification Stack)...")
    plot_reification_stack(df_pre, df_stage)
    
    print("Generating Tables...")
    generate_table_avg(df_pre, df_stage)
    generate_detailed_component_tables(df_pre, df_stage)
    
    print("Done.")

if __name__ == "__main__":
    main()