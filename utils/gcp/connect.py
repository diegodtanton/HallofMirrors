import subprocess
from .config import VM_NAME, ZONE

def connect():
    print(f"🔌 SSH Connecting to {VM_NAME}...")
    print("   (Type 'exit' to disconnect)")

    # Construct the command
    cmd = f"gcloud compute ssh {VM_NAME} --zone={ZONE}"

    try:
        # shell=True allows the SSH session to take over the terminal window interactively
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\nDisconnected.")

if __name__ == "__main__":
    connect()