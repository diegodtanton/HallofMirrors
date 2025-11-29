import os
import shutil
import glob

def get_project_root():
    # Assumes this script is located in /utils/
    # Returns the parent directory (the project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def clean_project():
    root = get_project_root()
    print(f"🧹 Cleaning project rooted at: {root}")

    # 1. Confirm with user to prevent accidental data loss
    confirm = input("WARNING: This will DELETE all training data, checkpoints, and logs.\nAre you sure? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # ------------------------------------------------
    # A. Clean __pycache__ (Recursive)
    # ------------------------------------------------
    print("\n--- Removing __pycache__ ---")
    cache_count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "__pycache__" in dirnames:
            path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(path)
                cache_count += 1
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
    print(f"Removed {cache_count} __pycache__ directories.")

    # ------------------------------------------------
    # B. Clean Artifact Directories (Data/Checkpoints/Logs)
    # ------------------------------------------------
    print("\n--- Removing Artifacts ---")
    # Folders to wipe content from, but keep the folder itself
    artifact_dirs = [
        "pretrain/checkpoints",
        "pretrain/data",
        "mirrors/checkpoints",
        "mirrors/data",
        "logs",
        "utils/output" # If you have one
    ]

    for d in artifact_dirs:
        full_path = os.path.join(root, d)
        if os.path.exists(full_path):
            try:
                # Remove the directory and everything in it
                shutil.rmtree(full_path)
                # Immediately recreate the empty directory
                os.makedirs(full_path, exist_ok=True)
                print(f"Cleaned: {d}/")
            except Exception as e:
                print(f"Error cleaning {d}: {e}")
        else:
            # Create it if it doesn't exist (clean slate)
            os.makedirs(full_path, exist_ok=True)

    # ------------------------------------------------
    # C. Remove Zip files (from previous deployments)
    # ------------------------------------------------
    print("\n--- Removing Archives ---")
    zip_files = glob.glob(os.path.join(root, "*.zip"))
    for z in zip_files:
        try:
            os.remove(z)
            print(f"Deleted: {os.path.basename(z)}")
        except Exception as e:
            print(f"Error deleting {z}: {e}")

    print("\nProject is clean and ready for deployment/fresh run.")

if __name__ == "__main__":
    clean_project()