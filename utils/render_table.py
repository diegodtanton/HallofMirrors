# ============================================
# File: utils/render_table.py
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "analysis_results"
CSV_PATH = os.path.join(OUTPUT_DIR, "table1_gauge_evolution.csv")

def render_table():
    if not os.path.exists(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 1. Clean up Index (Gauge Types)
    # They should already be cleaned by plot_story, but just in case
    # Assuming CSV first col is 'gauge_type'
    if 'gauge_type' in df.columns:
        df.set_index('gauge_type', inplace=True)
    
    # 2. Construct display strings "Mean ± Std"
    # Columns expected: Pre_Mean, Pre_Std, S1_Mean, S1_Std, ...
    
    phases = [
        ("Pretrain", "Pre"),
        ("Stage 1 (Rot)", "S1"),
        ("Stage 2 (Step)", "S2"),
        ("Stage 3 (Val)", "S3")
    ]
    
    display_df = pd.DataFrame(index=df.index)
    
    for label, prefix in phases:
        mean_col = f"{prefix}_Mean"
        std_col = f"{prefix}_Std"
        
        if mean_col in df.columns and std_col in df.columns:
            # Create formatted string
            display_df[label] = df.apply(
                lambda x: f"{x[mean_col]:.2f} ± {x[std_col]:.2f}", axis=1
            )
        else:
            display_df[label] = "-"

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        rowLabels=display_df.index,
        cellLoc='center',
        loc='center'
    )
    
    # Formatting
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0) # Stretch height
    
    # Styling headers
    for (i, j), cell in table.get_celld().items():
        if i == 0: # Header row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#f0f0f0')
        if j == -1: # Index col
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f9f9f9')

    plt.title("Table 1: Evolution of Gauge Scores (Mean ± Std)", weight='bold', y=1.05)
    
    save_path = os.path.join(OUTPUT_DIR, "table1_rendered.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Table image saved to {save_path}")

if __name__ == "__main__":
    render_table()