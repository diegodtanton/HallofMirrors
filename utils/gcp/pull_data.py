import os
import subprocess
import datetime
import sys
from .config import VM_NAME, ZONE, REMOTE_DIR, LOCAL_RESULTS_DIR

def pull_data():
    # 1. Create Timestamped Folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest_dir = os.path.join(LOCAL_RESULTS_DIR, timestamp)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"📥 Pulling data from {VM_NAME}...")
    print(f"📂 Destination: {dest_dir}")

    # 2. Define remote paths to grab
    # We grab the parent 'data' and 'checkpoints' folders for both phases
    targets = [
        f"{REMOTE_DIR}/pretrain/data",
        f"{REMOTE_DIR}/pretrain/checkpoints",
        f"{REMOTE_DIR}/mirrors/data",
        f"{REMOTE_DIR}/mirrors/checkpoints",
    ]

    # 3. Execute SCP commands
    for target in targets:
        # Determine local subdirectory structure
        # e.g. pretrain/data -> dest_dir/pretrain/data
        rel_path = target.replace(f"{REMOTE_DIR}/", "")
        local_target_path = os.path.join(dest_dir, os.path.dirname(rel_path))
        os.makedirs(local_target_path, exist_ok=True)

        cmd = [
            "gcloud", "compute", "scp",
            "--recurse",
            "--zone", ZONE,
            f"{VM_NAME}:{target}",
            local_target_path
        ]

        print(f"   Downloading: {rel_path}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            # It's common to fail if the folder doesn't exist yet on remote (e.g. no mirrors run yet)
            print(f"   ⚠️  Skipped (not found or empty): {rel_path}")
        else:
            print(f"   ✅ Success")

    print("\n✨ Download Complete!")
    print(f"   Explorer: {os.path.abspath(dest_dir)}")

if __name__ == "__main__":
    pull_data()