import os
import zipfile
import subprocess
import fnmatch
from .config import VM_NAME, ZONE, REMOTE_DIR, IGNORE_PATTERNS

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def should_ignore(path, root):
    rel_path = os.path.relpath(path, root)
    for part in rel_path.split(os.sep):
        for pattern in IGNORE_PATTERNS:
            if fnmatch.fnmatch(part, pattern):
                return True
    filename = os.path.basename(path)
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

def zip_project(zip_name="deploy.zip"):
    root = get_project_root()
    zip_path = os.path.join(root, zip_name)
    print(f"📦 Zipping project at {root}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder, _, files in os.walk(root):
            for file in files:
                full_path = os.path.join(folder, file)
                if should_ignore(full_path, root):
                    continue
                if file == zip_name:
                    continue
                rel_path = os.path.relpath(full_path, root)
                zipf.write(full_path, rel_path)
    return zip_path

def push_code():
    zip_path = zip_project()
    zip_filename = os.path.basename(zip_path)

    # 1. Clean Remote
    print(f"🧹 Force-cleaning remote directory: {REMOTE_DIR}...")
    remote_prep_cmd = (
        "sudo pkill -f python; "           
        f"sudo rm -rf {REMOTE_DIR}"        
    )
    subprocess.run([
        "gcloud", "compute", "ssh", VM_NAME,
        "--zone", ZONE,
        "--command", remote_prep_cmd
    ], shell=True) 

    # 2. Upload Zip
    print(f"rw Sending {zip_filename} to VM Home...")
    upload_cmd = [
        "gcloud", "compute", "scp",
        zip_path,
        f"{VM_NAME}:.",   
        "--zone", ZONE
    ]
    subprocess.run(upload_cmd, shell=True, check=True)

    # 3. Unzip, Install, Fix Permissions
    print("📂 Unzipping and installing requirements...")
    
    # THE FIX: Added --break-system-packages to the pip install command
    install_cmd = (
        f"sudo unzip -o {zip_filename} -d {REMOTE_DIR} && "
        f"sudo chown -R $USER:$USER {REMOTE_DIR} && "
        f"chmod +x {REMOTE_DIR}/launch_parallel.sh && "
        f"cd {REMOTE_DIR} && pip install -r requirements.txt --break-system-packages && "
        f"cd ~ && rm {zip_filename}"
    )

    subprocess.run([
        "gcloud", "compute", "ssh", VM_NAME,
        "--zone", ZONE,
        "--command", install_cmd
    ], shell=True, check=True)

    os.remove(zip_path)
    print("\n🚀 Deployment Complete!")
    print("   You can now run: ./launch_parallel.sh on the VM.")

if __name__ == "__main__":
    push_code()