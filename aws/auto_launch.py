"""
OrchestraNet — Auto-Launch EC2 Training Instance.

Retries launching until account verification completes, then:
1. Launches g4dn.xlarge with 150GB storage
2. Waits for instance to be running
3. Uploads project code via SCP
4. Runs setup + COCO download + training pipeline

Usage:
  python aws/auto_launch.py
"""

import json
import os
import subprocess
import sys
import time

# === Configuration ===
INSTANCE_TYPE = "g4dn.xlarge"
AMI_ID = "ami-07f2adfb18ee0bc01"
KEY_NAME = "orchestranet-training"
SG_ID = "sg-097452a2e2f3067b1"
KEY_PATH = os.path.expanduser("~/.ssh/orchestranet-training.pem")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRY_INTERVAL = 120  # seconds between retries
MAX_RETRIES = 30  # ~1 hour of retrying


def run(cmd, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        return None, result.stderr
    return result.stdout.strip(), None


def launch_instance():
    """Try to launch the EC2 instance."""
    cmd = (
        f'aws ec2 run-instances '
        f'--image-id {AMI_ID} '
        f'--instance-type {INSTANCE_TYPE} '
        f'--key-name {KEY_NAME} '
        f'--security-group-ids {SG_ID} '
        f'--block-device-mappings file://aws/bdm.json '
        f'--tag-specifications "ResourceType=instance,Tags=[{{Key=Name,Value=OrchestraNet-Training}}]" '
        f'--query "Instances[0].InstanceId" --output text'
    )
    return run(cmd, check=False)


def wait_for_running(instance_id):
    """Wait until instance is in 'running' state."""
    print(f"   Waiting for instance {instance_id} to be running...")
    cmd = f'aws ec2 wait instance-running --instance-ids {instance_id}'
    run(cmd)
    # Get public IP
    cmd = f'aws ec2 describe-instances --instance-ids {instance_id} --query "Reservations[0].Instances[0].PublicIpAddress" --output text'
    ip, _ = run(cmd)
    return ip


def wait_for_ssh(host, max_wait=300):
    """Wait until SSH is available."""
    print(f"   Waiting for SSH on {host}...")
    start = time.time()
    while time.time() - start < max_wait:
        result = subprocess.run(
            f'ssh -i "{KEY_PATH}" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@{host} echo ok',
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"   ✅ SSH connected!")
            return True
        time.sleep(10)
    return False


def upload_code(host):
    """Upload project code to the instance."""
    print(f"📤 Uploading project code to {host}...")
    cmd = (
        f'scp -i "{KEY_PATH}" -o StrictHostKeyChecking=no -r '
        f'-o "ExcludeFrom=/dev/null" '  # scp doesn't support exclude, we'll use tar
        f'"{PROJECT_DIR}" ubuntu@{host}:~/'
    )
    # Use tar to exclude unnecessary files
    tar_cmd = (
        f'cd "{os.path.dirname(PROJECT_DIR)}" && '
        f'tar czf - --exclude=__pycache__ --exclude=.git --exclude=data --exclude=checkpoints '
        f'--exclude=logs --exclude=exports --exclude="*.pyc" --exclude=.pytest_cache '
        f'--exclude="*.egg-info" CV_project | '
        f'ssh -i "{KEY_PATH}" -o StrictHostKeyChecking=no ubuntu@{host} "tar xzf - -C ~/"'
    )
    os.system(tar_cmd)
    print("   ✅ Code uploaded")


def start_training(host):
    """Start the training pipeline on the instance."""
    print("🚀 Starting training pipeline...")

    commands = [
        # Install dependencies
        'cd ~/CV_project && pip install -e ".[dev]" 2>&1 | tail -5',
        # Verify GPU
        'python3 -c "import torch; print(f\'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)\')" 2>&1',
        # Download COCO (in background)
        'cd ~/CV_project && bash aws/download_coco.sh ./data/coco > /tmp/coco_download.log 2>&1 &',
        # Wait for COCO download then start training in tmux
        'tmux new-session -d -s train "cd ~/CV_project && bash -c \'while [ ! -d data/coco/train2017 ]; do echo Waiting for COCO download...; sleep 30; done && echo COCO ready! && bash aws/train_full_pipeline.sh 2>&1 | tee training_output.log\'"',
        # Start S3 sync in background
        'tmux new-session -d -s sync "cd ~/CV_project && bash aws/sync_checkpoints.sh --watch"',
    ]

    for cmd in commands:
        ssh_cmd = f'ssh -i "{KEY_PATH}" -o StrictHostKeyChecking=no ubuntu@{host} "{cmd}"'
        print(f"  $ {cmd[:80]}...")
        os.system(ssh_cmd)

    print("")
    print("=" * 60)
    print("✅ Training launched!")
    print(f"   Instance IP: {host}")
    print(f"   SSH: ssh -i {KEY_PATH} ubuntu@{host}")
    print(f"   Monitor: ssh -i {KEY_PATH} ubuntu@{host} 'tmux attach -t train'")
    print("=" * 60)


def main():
    print("🎼 OrchestraNet — EC2 Auto-Launcher")
    print("=" * 50)
    print(f"   Instance: {INSTANCE_TYPE}")
    print(f"   AMI: {AMI_ID}")
    print(f"   Key: {KEY_PATH}")
    print("")

    # Retry loop for account verification
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"▶ Attempt {attempt}/{MAX_RETRIES}: Launching instance...")
        instance_id, err = launch_instance()

        if instance_id and not err:
            print(f"   ✅ Instance launched: {instance_id}")
            break
        elif err and "PendingVerification" in err:
            print(f"   ⏳ Account still verifying... retrying in {RETRY_INTERVAL}s")
            time.sleep(RETRY_INTERVAL)
        else:
            print(f"   ❌ Error: {err}")
            if "InsufficientInstanceCapacity" in (err or ""):
                print("   Trying a different AZ...")
                time.sleep(30)
            else:
                sys.exit(1)
    else:
        print("❌ Max retries exceeded. Check AWS account verification status.")
        sys.exit(1)

    # Wait for instance
    public_ip = wait_for_running(instance_id)
    print(f"   🌐 Public IP: {public_ip}")

    # Wait for SSH
    if not wait_for_ssh(public_ip):
        print("❌ SSH timeout. Instance may still be booting.")
        print(f"   Try manually: ssh -i {KEY_PATH} ubuntu@{public_ip}")
        sys.exit(1)

    # Upload code and start training
    upload_code(public_ip)
    start_training(public_ip)

    # Save instance info
    info = {
        "instance_id": instance_id,
        "public_ip": public_ip,
        "instance_type": INSTANCE_TYPE,
        "key_path": KEY_PATH,
    }
    info_path = os.path.join(PROJECT_DIR, "aws", "instance_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n📝 Instance info saved: {info_path}")


if __name__ == "__main__":
    main()
