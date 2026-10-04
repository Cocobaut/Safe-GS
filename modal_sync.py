"""
Modal sync & storage manager for Safe-GS (modal_sync.py)
=========================================================
Dedicated file for data management, environment verification, and synchronization:
  1. Check GPU and cloud volume connection.
  2. Upload Replica 8 Scene dataset to Modal Volume ('safe-gs-dataset') scene by scene.
  3. Upload outputs folder to Modal Volume ('safe-gs-outputs') (preserving 1:1 structure).
  4. Download results from Modal Volume to local machine.
  5. List existing dataset and output files on cloud volume.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure Windows console handles UTF-8 properly without charmap error
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import modal

# -----------------------------------------------------------------------------
# 1. Modal app & persistent volumes configuration
# -----------------------------------------------------------------------------
APP_NAME = "safe-gs-sync"
app = modal.App(APP_NAME)

# Volume for Replica dataset
dataset_vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)

# Volume for outputs results
outputs_vol = modal.Volume.from_name("safe-gs-outputs", create_if_missing=True)

# Lightweight GPU image for checking connection
check_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "torch", index_url="https://download.pytorch.org/whl/cu121")
)


# -----------------------------------------------------------------------------
# 2. Check GPU connection & cloud volumes
# -----------------------------------------------------------------------------
@app.function(
    image=check_image,
    gpu="A10G",
    volumes={
        "/data": dataset_vol,
        "/outputs": outputs_vol,
    },
    timeout=180,
)
def check_connection():
    """Launch a GPU worker to verify NVIDIA GPU and check both volumes."""
    import torch

    print("=" * 60)
    print(">>> Checking modal gpu and volume connection for safe-gs <<<")
    print("=" * 60)

    # 1. Check GPU
    print(f"Cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Gpu: {gpu_name} ({vram_gb:.2f} gb vram)")
        print(f"Cuda version: {torch.version.cuda}")

    # 2. Check dataset volume
    print("\n--- Checking dataset volume ('safe-gs-dataset' mounted at /data) ---")
    dataset_vol.reload()
    replica_dir = Path("/data/Replica 8 Scene/Replica")
    if replica_dir.exists():
        scenes = sorted([p.name for p in replica_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
        print(f"[SUCCESS] Found replica directory ({len(scenes)} scenes): {scenes}")
    else:
        print("[WARNING] Dataset not found at '/data/Replica 8 Scene/Replica'.")
        print("    Run this command locally to upload: python modal_sync.py upload-data")

    # 3. Check outputs volume
    print("\n--- Checking outputs volume ('safe-gs-outputs' mounted at /outputs) ---")
    outputs_vol.reload()
    outputs_dir = Path("/outputs")
    if outputs_dir.exists():
        items = sorted([p.name for p in outputs_dir.iterdir()])
        print(f"[SUCCESS] Outputs directory is ready. Current content: {items}")
    else:
        print("[INFO] Outputs directory is empty, ready to receive task results.")
    print("=" * 60)


@app.function(
    image=modal.Image.debian_slim(python_version="3.10"),
    volumes={"/outputs": outputs_vol},
    timeout=120,
)
def list_outputs_remote():
    """List entire outputs directory tree on cloud volume."""
    outputs_vol.reload()
    base = Path("/outputs")
    print("\n=== Content of modal volume: safe-gs-outputs ===")
    if not base.exists():
        print("[INFO] Outputs directory does not exist on volume yet.")
        return

    count = 0
    for root, dirs, files in os.walk(base):
        rel = os.path.relpath(root, base)
        indent = "  " * rel.count(os.sep) if rel != "." else ""
        folder_name = os.path.basename(root) if rel != "." else "outputs"
        print(f"{indent}[D] {folder_name}/")
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            size_str = f"{size / 1024**2:.2f} mb" if size > 1024**2 else f"{size / 1024:.1f} kb"
            print(f"{indent}  - {f} ({size_str})")
            count += 1
    print(f"\nTotal files: {count}\n" + "=" * 60)


# -----------------------------------------------------------------------------
# 3. Local upload & download functions (run directly from your machine)
# -----------------------------------------------------------------------------
def upload_replica_data(local_dir: str = "../Replica 8 Scene", scene: str = None):
    """
    Upload Replica 8 Scene dataset from local machine to Modal Volume 'safe-gs-dataset'.
    Uploads scene by scene and confirms completion only after data is committed on cloud.
    """
    vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
    local_p = Path(local_dir)
    if not local_p.exists():
        alt = Path("F:/Replica 8 Scene")
        if alt.exists():
            local_p = alt
        else:
            print(f"[ERROR] Dataset path not found at '{local_p}' (or '{alt}')")
            return

    replica_p = local_p / "Replica" if (local_p / "Replica").exists() else local_p
    print(f"[INFO] Local dataset directory: {replica_p}")

    # Step 1: Upload general parameter files (cam_params.json, mesh .ply files)
    print("\n[STEP 1/2] Uploading base parameter files (cam_params.json, mesh.ply)...")
    param_files = [f for f in replica_p.glob("*.*") if f.is_file() and not f.name.startswith(".")]
    if param_files:
        with vol.batch_upload(force=True) as batch:
            for f in param_files:
                remote_f = f"/Replica 8 Scene/Replica/{f.name}"
                batch.put_file(f, remote_f)
        print(f"[SUCCESS] Saved {len(param_files)} base parameter file(s) to cloud volume!")
    else:
        print("[INFO] No base parameter files found to upload.")

    # Step 2: Upload scene directories
    all_scenes = [d.name for d in replica_p.iterdir() if d.is_dir() and not d.name.startswith(".")]
    target_scenes = [scene] if scene else sorted(all_scenes)

    print(f"\n[STEP 2/2] Preparing to upload {len(target_scenes)} scene(s): {target_scenes}")
    for idx, sc in enumerate(target_scenes, 1):
        sc_dir = replica_p / sc
        if not sc_dir.exists():
            print(f"[WARNING] Skipping '{sc}' because directory was not found at {sc_dir}")
            continue

        files_count = sum(1 for _ in sc_dir.rglob("*") if _.is_file())
        size_mb = sum(f.stat().st_size for f in sc_dir.rglob("*") if f.is_file()) / (1024 * 1024)

        print(f"\n--> [{idx}/{len(target_scenes)}] Uploading scene '{sc}' ({files_count} files, {size_mb:.1f} mb)...")
        print("    (Please keep network connected, modal is transferring data to cloud volume...)")

        with vol.batch_upload(force=True) as batch:
            batch.put_directory(sc_dir, f"/Replica 8 Scene/Replica/{sc}")

        print(f"[SUCCESS] Saved scene '{sc}' to cloud volume successfully!")

    print("\n" + "=" * 60)
    print("[SUCCESS] Dataset upload process completed successfully!")
    print("=" * 60)


def upload_outputs(local_dir: str = "outputs", target: str = "Replica", subfolder: str = None):
    """
    Upload local outputs directory to Modal Volume 'safe-gs-outputs'.
    By default uploads ONLY the 'Replica' directory inside outputs (skipping DTU).
    Preserves 1:1 directory structure.
    """
    vol = modal.Volume.from_name("safe-gs-outputs", create_if_missing=True)
    local_p = Path(local_dir)
    if not local_p.exists():
        alt = Path("F:/Safe-GS/outputs")
        if alt.exists():
            local_p = alt
        else:
            print(f"[ERROR] Outputs directory not found at '{local_p}' (or '{alt}')")
            return

    target_dir = local_p / target
    if not target_dir.exists():
        print(f"[ERROR] Directory '{target}' not found inside '{local_p}'.")
        return

    print(f"[INFO] Uploading directory '{target}' inside outputs to modal volume 'safe-gs-outputs' (skipping dtu)...")

    if subfolder:
        sub_items = [target_dir / subfolder] if (target_dir / subfolder).exists() else []
        if not sub_items:
            print(f"[ERROR] Subfolder '{subfolder}' not found in {target_dir}")
            return
    else:
        sub_items = sorted([it for it in target_dir.iterdir() if not it.name.startswith(".")])

    print(f"[INFO] List of folders to upload ({len(sub_items)}): {[it.name for it in sub_items]}")

    for idx, item in enumerate(sub_items, 1):
        remote_path = f"/{target}/{item.name}"
        if item.is_dir():
            files_count = sum(1 for _ in item.rglob("*") if _.is_file())
            size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"\n--> [{idx}/{len(sub_items)}] Uploading '{target}/{item.name}' ({files_count} files, {size_mb:.1f} mb)...")
            print(f"    (Uploading to cloud at '{remote_path}', please keep network connected...)")
            with vol.batch_upload(force=True) as batch:
                batch.put_directory(item, remote_path)
            print(f"[SUCCESS] Saved '{target}/{item.name}' to cloud volume successfully!")
        else:
            with vol.batch_upload(force=True) as batch:
                batch.put_file(item, remote_path)
            print(f"[SUCCESS] Saved file '{target}/{item.name}' to cloud volume successfully!")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Finished uploading outputs/{target.lower()} to modal volume!")
    print("=" * 60)


def download_outputs(remote_path: str = "Replica", local_dir: str = "outputs/Replica"):
    """
    Download data from Modal Volume 'safe-gs-outputs' to local machine.
    Example: remote_path='Replica' -> local_dir='outputs/Replica'
    """
    os.makedirs(Path(local_dir).parent, exist_ok=True)
    print(f"[INFO] Downloading '{remote_path}' from cloud volume to '{local_dir}'...")
    cmd = [sys.executable, "-m", "modal", "volume", "get", "safe-gs-outputs", remote_path, local_dir]
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Download completed successfully to: {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed with exit code {e.returncode}.")


def list_dataset_local():
    """List available scenes and parameter files on Volume safe-gs-dataset."""
    vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
    print("\n=== Content of modal volume: safe-gs-dataset ===")
    try:
        entries = list(vol.iterdir("/Replica 8 Scene/Replica"))
        if not entries:
            print("[INFO] No data found in /Replica 8 Scene/Replica.")
            return
        scenes = []
        files = []
        for e in entries:
            name = Path(e.path).name
            if e.type.name == "DIRECTORY":
                scenes.append(name)
            else:
                files.append(name)
        print(f"[SUCCESS] Parameter files: {files}")
        print(f"[SUCCESS] Available scenes ({len(scenes)}): {sorted(scenes)}")
    except Exception as ex:
        print(f"[ERROR] Inspection failed: {ex}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# 4. CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal sync and storage manager for safe-gs")
    subparsers = parser.add_subparsers(dest="command")

    # Command upload-data
    p_data = subparsers.add_parser("upload-data", help="Upload replica 8 scene dataset to modal volume")
    p_data.add_argument("--dir", default="../Replica 8 Scene", help="Path to replica 8 scene directory (default: ../Replica 8 Scene)")
    p_data.add_argument("--scene", default=None, help="Specify a single scene to upload (e.g., office0)")

    # Command upload-outputs
    p_out = subparsers.add_parser("upload-outputs", help="Upload outputs directory to modal volume")
    p_out.add_argument("--dir", default="outputs", help="Path to outputs directory (default: outputs)")
    p_out.add_argument("--target", default="Replica", help="Target folder to upload (default: Replica)")
    p_out.add_argument("--subfolder", default=None, help="Upload a specific subfolder inside replica (e.g., sfm, bounding_box_3D)")

    # Command download-outputs
    p_down = subparsers.add_parser("download-outputs", help="Download results from modal volume to local machine")
    p_down.add_argument("--remote", default="Replica", help="Remote path on volume (default: Replica)")
    p_down.add_argument("--local", default="outputs/Replica", help="Local destination path (default: outputs/Replica)")

    # Command list-outputs
    subparsers.add_parser("list-outputs", help="List outputs files on cloud volume")

    # Command list-dataset
    subparsers.add_parser("list-dataset", help="List dataset scenes on cloud volume")

    args = parser.parse_args()

    if args.command == "upload-data":
        upload_replica_data(args.dir, args.scene)
    elif args.command == "upload-outputs":
        upload_outputs(args.dir, args.target, args.subfolder)
    elif args.command == "download-outputs":
        download_outputs(args.remote, args.local)
    elif args.command == "list-outputs":
        list_outputs_remote.remote()
    elif args.command == "list-dataset":
        list_dataset_local()
    else:
        print("\n=== Modal sync and storage manager ===")
        print("1. Upload data to cloud:")
        print("   python modal_sync.py upload-data                  # Upload all 8 scenes of replica")
        print("   python modal_sync.py upload-data --scene office0  # Upload only scene 1 (office0, ~1.3gb)")
        print("   python modal_sync.py upload-outputs               # Upload replica directory inside outputs (skipping dtu)")
        print("   python modal_sync.py upload-outputs --subfolder sfm # Upload only sfm subfolder")
        print("\n2. Check cloud volume data:")
        print("   python modal_sync.py list-dataset                 # Check uploaded dataset scenes on cloud")
        print("   python modal_sync.py list-outputs                 # Check outputs results on cloud")
        print("   python -m modal run modal_sync.py::check_connection")
        print("\n3. Download results to local machine:")
        print("   python modal_sync.py download-outputs\n")
