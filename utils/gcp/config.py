import os

# ====================================================
# GCP CONFIGURATION
# ====================================================

VM_NAME = "mirrors-experiment-01"
ZONE = "us-central1-f"  # Ensure this matches your VM's zone
REMOTE_DIR = "gauges_experiment" # Where code lives on the VM

# Local folder where we save downloaded results
LOCAL_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "downloaded_results")

# Port for the view.py visualization
PORT = 8080

# Patterns to ignore when Zipping code for deployment
IGNORE_PATTERNS = [
    "__pycache__",
    "*.git*",
    "*.vscode*",
    "venv",
    "env",
    "logs",
    "downloaded_results",
    "deploy.zip",
    "*.DS_Store",
    "utils/gcp"
]