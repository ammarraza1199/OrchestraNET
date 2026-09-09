"""
OrchestraNet — AWS EC2 Training Launcher.

Python script to manage EC2 instances for remote training:
  1. Upload project code to a running EC2 instance
  2. Start training via SSH
  3. Monitor training progress
  4. Download results when complete

Prerequisites:
  - AWS CLI configured (`aws configure`)
  - SSH key for the EC2 instance
  - EC2 instance already launched (this script doesn't create instances)

Usage:
  # Upload code and start training
  python aws/launch_training.py --host <ec2-ip> --key ~/.ssh/my-key.pem --start

  # Monitor training
  python aws/launch_training.py --host <ec2-ip> --key ~/.ssh/my-key.pem --monitor

  # Download results
  python aws/launch_training.py --host <ec2-ip> --key ~/.ssh/my-key.pem --download
"""

import argparse
import os
import subprocess
import sys
import time


def run_ssh(host: str, key: str, command: str, user: str = "ubuntu") -> int:
    """Run a command on the remote EC2 instance via SSH."""
    ssh_cmd = [
        "ssh", "-i", key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{user}@{host}",
        command,
    ]
    print(f"  $ {command}")
    result = subprocess.run(ssh_cmd, capture_output=False)
    return result.returncode


def upload_code(host: str, key: str, user: str = "ubuntu"):
    """Upload project code to EC2 instance using rsync."""
    print("📤 Uploading project code...")

    rsync_cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {key} -o StrictHostKeyChecking=no",
        "--exclude", "__pycache__",
        "--exclude", ".git",
        "--exclude", "data",
        "--exclude", "checkpoints",
        "--exclude", "logs",
        "--exclude", "exports",
        "--exclude", "*.pyc",
        "--exclude", ".pytest_cache",
        "--exclude", "*.egg-info",
        "./",
        f"{user}@{host}:~/CV_project/",
    ]
    subprocess.run(rsync_cmd, check=True)
    print("   ✅ Code uploaded")


def start_training(host: str, key: str, user: str = "ubuntu"):
    """Start the full training pipeline on EC2."""
    print("🚀 Starting training pipeline...")

    # Start in a tmux session so it survives SSH disconnect
    commands = [
        # Setup (idempotent)
        "cd ~/CV_project && bash aws/setup_instance.sh",
        # Download COCO if needed
        "cd ~/CV_project && bash aws/download_coco.sh",
        # Start training in tmux
        "tmux new-session -d -s train 'cd ~/CV_project && source ~/orchestranet_env/bin/activate && bash aws/train_full_pipeline.sh 2>&1 | tee training_output.log'",
        # Start S3 sync in background
        "tmux new-session -d -s sync 'cd ~/CV_project && source ~/orchestranet_env/bin/activate && bash aws/sync_checkpoints.sh --watch'",
    ]

    for cmd in commands:
        ret = run_ssh(host, key, cmd, user)
        if ret != 0:
            print(f"⚠️  Command returned {ret}, continuing...")

    print("")
    print("✅ Training started in tmux session 'train'")
    print("   S3 sync running in tmux session 'sync'")
    print("")
    print("To monitor:")
    print(f"  ssh -i {key} {user}@{host}")
    print("  tmux attach -t train")


def monitor_training(host: str, key: str, user: str = "ubuntu"):
    """Check training progress on EC2."""
    print("📊 Checking training progress...\n")

    # Check if training is running
    run_ssh(host, key, "tmux has-session -t train 2>/dev/null && echo '✅ Training is RUNNING' || echo '❌ Training is NOT running'", user)

    # Show GPU status
    print("\n🖥️  GPU Status:")
    run_ssh(host, key, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader", user)

    # Show latest training output
    print("\n📋 Latest training output:")
    run_ssh(host, key, "cd ~/CV_project && tail -20 training_output.log 2>/dev/null || echo 'No log file yet'", user)

    # Show checkpoint directory
    print("\n💾 Checkpoints:")
    run_ssh(host, key, "ls -lhrt ~/CV_project/checkpoints/*.pt 2>/dev/null | tail -5 || echo 'No checkpoints yet'", user)

    # Show disk usage
    print("\n💿 Disk Usage:")
    run_ssh(host, key, "df -h / | tail -1", user)


def download_results(host: str, key: str, local_dir: str = ".", user: str = "ubuntu"):
    """Download training results from EC2."""
    print("📥 Downloading results...")

    for remote_dir, local_subdir in [
        ("checkpoints", "checkpoints"),
        ("logs", "logs"),
        ("results", "results"),
        ("exports", "exports"),
    ]:
        local_path = os.path.join(local_dir, local_subdir)
        os.makedirs(local_path, exist_ok=True)

        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "-e", f"ssh -i {key} -o StrictHostKeyChecking=no",
            f"{user}@{host}:~/CV_project/{remote_dir}/",
            local_path + "/",
        ]
        print(f"\n  Downloading {remote_dir}/...")
        subprocess.run(rsync_cmd)

    print("\n✅ Results downloaded")


def main():
    parser = argparse.ArgumentParser(description="OrchestraNet EC2 Training Launcher")
    parser.add_argument("--host", required=True, help="EC2 public IP or hostname")
    parser.add_argument("--key", required=True, help="SSH private key path")
    parser.add_argument("--user", default="ubuntu", help="SSH username")

    # Actions
    parser.add_argument("--upload", action="store_true", help="Upload code to EC2")
    parser.add_argument("--start", action="store_true", help="Upload + start training")
    parser.add_argument("--monitor", action="store_true", help="Check training progress")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--local-dir", default=".", help="Local directory for downloads")

    args = parser.parse_args()

    if args.upload or args.start:
        upload_code(args.host, args.key, args.user)

    if args.start:
        start_training(args.host, args.key, args.user)

    if args.monitor:
        monitor_training(args.host, args.key, args.user)

    if args.download:
        download_results(args.host, args.key, args.local_dir, args.user)

    if not any([args.upload, args.start, args.monitor, args.download]):
        print("No action specified. Use --upload, --start, --monitor, or --download")
        parser.print_help()


if __name__ == "__main__":
    main()
