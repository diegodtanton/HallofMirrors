import subprocess
import time
import webbrowser
from .config import VM_NAME, ZONE, REMOTE_DIR, PORT

def view_training():
    print(f"🔭 Preparing to view {VM_NAME}...")

    # 1. Start the Remote HTTP Server (Background)
    # We send a command to start the python server using 'nohup' so it keeps running
    print(f"   📡 Restarting remote web server on port {PORT}...")
    start_server_cmd = [
        "gcloud", "compute", "ssh", VM_NAME,
        "--zone", ZONE,
        "--command",
        f"cd {REMOTE_DIR} && nohup python3 -m http.server {PORT} > /dev/null 2>&1 &"
    ]
    subprocess.run(start_server_cmd, shell=True)

    # 2. Establish Tunnel
    print(f"   fw Opening tunnel: Local {PORT} -> Remote {PORT}")
    print("   Press CTRL+C to stop viewing.\n")

    url = f"http://localhost:{PORT}"
    
    # --ssh-flag syntax ensures compatibility with Windows PowerShell
    tunnel_cmd = [
        "gcloud", "compute", "ssh", VM_NAME,
        "--zone", ZONE,
        f"--ssh-flag=-N",
        f"--ssh-flag=-L {PORT}:localhost:{PORT}"
    ]

    try:
        # Launch the tunnel
        proc = subprocess.Popen(tunnel_cmd, shell=True)
        
        # Give the tunnel a moment to shake hands
        time.sleep(3) 
        
        print(f"✅ Active! Opening {url} ...")
        webbrowser.open(url)
        
        # Keep script running until user kills it
        proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Closing tunnel.")
        proc.terminate()

if __name__ == "__main__":
    view_training()