#!/usr/bin/env python3
"""
OrchestraNet — Master Backup & Google Drive Sync Utility.

Bundles all critical research artifacts:
  - Checkpoints (.pt weights: orchestranet_epoch34.pt, router_trained.pt, etc.)
  - Paper Tables (LaTeX .tex, Markdown .md, and paper_metrics_summary.json)
  - Training Logs (pipeline logs, individual micro-model logs, tensorboard)
  - Evaluation Results (JSON result files)

Automatically detects Google Drive mount points and syncs everything.

Usage:
  python scripts/package_and_sync.py
"""

import datetime
import os
import shutil
import sys
import tarfile
from pathlib import Path


def main():
    print("=" * 65)
    print("📦 OrchestraNet — Master Backup & Google Drive Sync")
    print("=" * 65)

    base_dir = Path(__file__).parent.parent.resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"OrchestraNet_PhD_Backup_{timestamp}"
    staging_dir = Path(f"/root/{backup_name}")

    os.makedirs(staging_dir, exist_ok=True)
    print(f"📁 Staging folder created at: {staging_dir}")

    # 1. Collect Key Directories
    components = [
        ("paper_tables", base_dir / "paper_tables", "All LaTeX & Markdown publication tables"),
        ("checkpoints", base_dir / "checkpoints", "Trained model weights (.pt files)"),
        ("results", base_dir / "results", "Evaluation metrics and JSON results"),
        ("logs", base_dir / "logs", "Training logs, TensorBoard events, diagnostic reports"),
    ]

    total_files = 0
    total_bytes = 0

    for name, src, desc in components:
        dst = staging_dir / name
        if src.exists():
            print(f"\n  ▶ Copying {name} ({desc})...")
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                count = sum(1 for _ in dst.rglob("*") if _.is_file())
                size = sum(_.stat().st_size for _ in dst.rglob("*") if _.is_file())
                print(f"    ✅ Copied {count} files ({size / (1024*1024):.2f} MB)")
                total_files += count
                total_bytes += size
        else:
            print(f"  ⚠️  Notice: {src} not found, skipping.")

    # Also collect any loose checkpoint files or logs in root
    for pt in base_dir.glob("*.pt"):
        shutil.copy2(pt, staging_dir / "checkpoints" / pt.name)
        total_files += 1
        total_bytes += pt.stat().st_size

    # 2. Create Master Compressed Archive (.tar.gz)
    archive_path = Path(f"/root/{backup_name}.tar.gz")
    print(f"\n🗜️  Compressing into master archive: {archive_path} ...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(staging_dir, arcname=backup_name)
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Master archive created: {archive_path.name} ({archive_size_mb:.2f} MB)")

    # 3. Detect and Sync to Google Drive
    drive_candidates = [
        Path("/root/drive"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive"),
        Path.home() / "drive",
        Path.home() / "GoogleDrive",
    ]

    synced_target = None
    for cand in drive_candidates:
        if cand.exists() and cand.is_dir():
            synced_target = cand
            break

    print("\n" + "=" * 65)
    if synced_target:
        print(f"🔗 Google Drive detected at: {synced_target}")
        dest_folder = synced_target / "OrchestraNet_Backup"
        os.makedirs(dest_folder, exist_ok=True)

        print("  ▶ Copying uncompressed folders to Google Drive...")
        shutil.copytree(staging_dir, dest_folder / backup_name, dirs_exist_ok=True)

        print("  ▶ Copying master .tar.gz archive to Google Drive...")
        shutil.copy2(archive_path, dest_folder / archive_path.name)

        print(f"\n🎉 SUCCESS: All checkpoints, tables, and logs synced to Google Drive!")
        print(f"   Destination: {dest_folder}")
    else:
        print("⚠️  Google Drive is not mounted yet.")
        print(f"   All files have been safely staged and archived locally at:")
        print(f"   1. Uncompressed folder: {staging_dir}")
        print(f"   2. Master Archive:     {archive_path} ({archive_size_mb:.2f} MB)")
        print("\n💡 To mount Google Drive, follow the step-by-step guide below.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
