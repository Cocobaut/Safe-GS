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
        print("[WARNING] Replica dataset not found at '/data/Replica 8 Scene/Replica'.")
        print("    Run this command locally to upload: python modal_sync.py upload-data")

    spinnerf_dir = Path("/data/SPIn-NeRF")
    if spinnerf_dir.exists():
        sp_items = sorted([p.name for p in spinnerf_dir.iterdir() if not p.name.startswith(".")])
        print(f"[SUCCESS] Found SPIn-NeRF directory ({len(sp_items)} items): {sp_items}")
    else:
        print("[INFO] SPIn-NeRF not found on volume yet. Upload via: python modal_sync.py upload-spinnerf")

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


@app.function(
    image=check_image,
    volumes={"/data": dataset_vol},
    timeout=120,
)
def clean_spinnerf_cloud_data():
    """Xóa sạch thư mục /data/SPIn-NeRF cũ trên Modal Volume 'safe-gs-dataset' để giải phóng dung lượng."""
    import shutil
    print("=" * 60)
    print(">>> DỌN DẸP /data/SPIn-NeRF TRÊN MODAL DATASET VOLUME <<<")
    print("=" * 60)
    dataset_vol.reload()
    target_path = Path("/data/SPIn-NeRF")
    if target_path.exists():
        sub_items = [p.name for p in target_path.iterdir()]
        print(f"[*] Tìm thấy /data/SPIn-NeRF hiện có các mục cũ: {sub_items}")
        shutil.rmtree(str(target_path), ignore_errors=True)
        print("[✓] Đã xóa toàn bộ thư mục /data/SPIn-NeRF cũ.")
    else:
        print("[*] Thư mục /data/SPIn-NeRF chưa tồn tại, không cần xóa.")

    dataset_vol.commit()
    print("[✓] Đã commit thay đổi lên Modal Volume 'safe-gs-dataset'!")


@app.function(
    image=check_image,
    volumes={"/outputs": outputs_vol},
    timeout=120,
)
def clean_splatfill_cloud_output(scene: str = "1"):
    """Xóa sạch thư mục kết quả splatfill/{scene} trên Modal Volume 'safe-gs-outputs'."""
    import shutil
    print("=" * 60)
    print(f">>> DỌN DẸP /outputs/splatfill/{scene} TRÊN MODAL OUTPUTS VOLUME <<<")
    print("=" * 60)
    outputs_vol.reload()
    
    deleted = False
    targets = [
        Path(f"/outputs/splatfill/{scene}"),
        Path(f"/outputs/{scene}"),
    ]
    for p in targets:
        if p.exists():
            print(f"[*] Tìm thấy {p}, đang tiến hành xóa...")
            shutil.rmtree(str(p), ignore_errors=True)
            print(f"[✓] Đã xóa {p}.")
            deleted = True
            
    if not deleted:
        print(f"[*] Không tìm thấy kết quả cũ của scene '{scene}', không cần xóa.")
        
    outputs_vol.commit()
    print("[✓] Đã commit thay đổi lên Modal Volume 'safe-gs-outputs'!")


def upload_spinnerf_data(local_dir: str = None, scene: str = None):
    """
    Upload 10 scenes SPIn-NeRF từ F:/SPIn-NeRF/spinnerf-dataset lên Modal Volume 'safe-gs-dataset'
    dưới thư mục chuẩn '/SPIn-NeRF/{scene}' (ví dụ: /SPIn-NeRF/1, /SPIn-NeRF/book,...).
    """
    vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)

    candidates = [
        Path(local_dir) if local_dir else None,
        Path("../SPIn-NeRF/spinnerf-dataset"),
        Path("F:/SPIn-NeRF/spinnerf-dataset"),
        Path("../SPIn-NeRF"),
        Path("F:/SPIn-NeRF"),
    ]
    local_p = None
    for cand in candidates:
        if cand and cand.exists():
            if (cand / "spinnerf-dataset").exists():
                local_p = cand / "spinnerf-dataset"
            else:
                local_p = cand
            break

    if not local_p or not local_p.exists():
        print(f"[ERROR] Không tìm thấy thư mục SPIn-NeRF dataset trên máy tính!")
        return

    print(f"[INFO] Thư mục SPIn-NeRF dataset nguồn: {local_p}", flush=True)

    # Danh sách 10 scenes
    all_scenes = [d.name for d in local_p.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
    target_scenes = [scene] if scene else sorted(all_scenes)

    print(f"\nChuẩn bị upload {len(target_scenes)} scene(s) lên Modal Volume (/SPIn-NeRF/): {target_scenes}", flush=True)
    for idx, sc_name in enumerate(target_scenes, 1):
        sc_path = local_p / sc_name
        if not sc_path.exists():
            print(f"[WARNING] Bỏ qua '{sc_name}' vì không tìm thấy tại {sc_path}", flush=True)
            continue

        files_count = sum(1 for _ in sc_path.rglob("*") if _.is_file())
        size_mb = sum(f.stat().st_size for f in sc_path.rglob("*") if f.is_file()) / (1024 * 1024)
        remote_dest = f"/SPIn-NeRF/{sc_name}"

        print(f"\n--> [{idx}/{len(target_scenes)}] Đang tải scene '{sc_name}' ({files_count} files, {size_mb:.1f} MB)...", flush=True)
        print(f"    Đích đến trên Cloud: {remote_dest}", flush=True)

        with vol.batch_upload(force=True) as batch:
            batch.put_directory(sc_path, remote_dest)

        print(f"[SUCCESS] Đã lưu scene '{sc_name}' vào {remote_dest} thành công!", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("[SUCCESS] Hoàn tất quá trình upload SPIn-NeRF lên Modal Volume!", flush=True)
    print("=" * 60)


def upload_lerf_data(scene: str, local_dir: str = "F:/LERF Datasets", prompts: str = None, prompts_only: bool = False):
    """
    Upload 1 scene LERF (figurines / teatime) lên Modal Volume 'safe-gs-dataset' tại '/LERF/{scene}':
    chỉ images/ (độ phân giải gốc) + transforms.json + prompts.json (điểm click chọn vật, cho SAM2).
    """
    vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
    sc_path = Path(local_dir) / scene
    if (sc_path / scene / "transforms.json").exists():  # cấu trúc giải nén LERF: <scene>/<scene>/...
        sc_path = sc_path / scene
    if not (sc_path / "transforms.json").exists():
        print(f"[ERROR] Không thấy transforms.json trong {sc_path}")
        return

    prompts_p = Path(prompts) if prompts else Path("outputs/lerf_prompts") / f"{scene}.json"
    remote_dest = f"/LERF/{scene}"
    with vol.batch_upload(force=True) as batch:
        if not prompts_only:
            n_imgs = len(list((sc_path / "images").glob("*")))
            print(f"[INFO] Upload {sc_path} -> {remote_dest} ({n_imgs} ảnh + transforms.json)...", flush=True)
            batch.put_directory(sc_path / "images", f"{remote_dest}/images")
            batch.put_file(sc_path / "transforms.json", f"{remote_dest}/transforms.json")
        if prompts_p.exists():
            batch.put_file(prompts_p, f"{remote_dest}/prompts.json")
            print(f"[INFO] Kèm prompts SAM2: {prompts_p}", flush=True)
        else:
            print(f"[WARNING] Chưa có {prompts_p} (chạy scripts/lerf_select_object.py --scene {scene} để tạo)", flush=True)
    print(f"[SUCCESS] Đã upload scene LERF '{scene}' vào {remote_dest}!", flush=True)


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


def download_outputs(target: str = "splatfill", scene: str = None, local_dir: str = None, remote_path: str = None):
    """
    Download outputs from Modal Volume 'safe-gs-outputs' to local machine.
    - Default target 'splatfill' downloads directly into 'Based_Model/SplatFill/output'
    - Default target 'Replica' downloads into 'outputs/Replica'
    """
    actual_target = remote_path if remote_path else target
    if local_dir is None:
        if actual_target.lower().startswith("splatfill"):
            local_dir = "Based_Model/SplatFill/output"
        else:
            local_dir = f"outputs/{actual_target}"

    local_base = Path(local_dir)
    local_base.mkdir(parents=True, exist_ok=True)

    vol = modal.Volume.from_name("safe-gs-outputs", create_if_missing=True)
    remote_base = actual_target
    if scene and not remote_base.endswith(scene):
        remote_base = f"{actual_target}/{scene}"

    print(f"\n[INFO] Đang quét kết quả từ Modal Volume 'safe-gs-outputs' (thư mục: '{remote_base}')...")
    try:
        entries = [e for e in vol.iterdir(remote_base) if e.type.name == "FILE"]
    except Exception as e:
        print(f"[ERROR] Không thể duyệt dữ liệu tại '{remote_base}' trên Volume: {e}")
        return

    if not entries:
        print(f"[INFO] Thư mục '{remote_base}' trên Cloud hiện chưa có file kết quả nào.")
        return

    print(f"[INFO] Tìm thấy {len(entries)} files. Đang tải về: {local_base.resolve()}...")
    for idx, entry in enumerate(entries, 1):
        rel = Path(entry.path)
        if rel.parts[0] == actual_target:
            rel = Path(*rel.parts[1:])
        
        # Chuẩn hóa nếu có lặp scene (ví dụ 1/1/model.ply -> 1/model.ply)
        parts = rel.parts
        if len(parts) >= 2 and parts[0] == parts[1]:
            rel = Path(parts[0], *parts[2:])

        dest_file = local_base / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        size_mb = entry.size / (1024 * 1024)
        print(f"  [{idx}/{len(entries)}] Tải {rel} ({size_mb:.2f} MB)...", end="", flush=True)

        with open(dest_file, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        print(" [XONG]")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Đã tải toàn bộ kết quả về: {local_base.resolve()}")
    print("=" * 60 + "\n")


def list_dataset_local():
    """List available scenes and parameter files on Volume safe-gs-dataset."""
    vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
    print("\n=== Content of modal volume: safe-gs-dataset ===")
    
    # 1. Replica
    try:
        entries = list(vol.iterdir("/Replica 8 Scene/Replica"))
        if entries:
            scenes = []
            files = []
            for e in entries:
                name = Path(e.path).name
                if e.type.name == "DIRECTORY":
                    scenes.append(name)
                else:
                    files.append(name)
            print(f"[SUCCESS] Replica base files: {files}")
            print(f"[SUCCESS] Replica scenes ({len(scenes)}): {sorted(scenes)}")
        else:
            print("[INFO] No data found in /Replica 8 Scene/Replica.")
    except Exception as ex:
        print(f"[INFO] Replica inspection: {ex}")

    # 2. SPIn-NeRF
    try:
        sp_entries = list(vol.iterdir("/SPIn-NeRF"))
        if sp_entries:
            sp_items = [Path(e.path).name for e in sp_entries]
            print(f"[SUCCESS] SPIn-NeRF items ({len(sp_items)}): {sorted(sp_items)}")
        else:
            print("[INFO] No data found in /SPIn-NeRF.")
    except Exception as ex:
        print(f"[INFO] SPIn-NeRF inspection: {ex}")

    print("=" * 60)


# -----------------------------------------------------------------------------
# 4. CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal sync and storage manager for safe-gs")
    subparsers = parser.add_subparsers(dest="command")

    # Command upload-data (Replica)
    p_data = subparsers.add_parser("upload-data", help="Upload replica 8 scene dataset to modal volume")
    p_data.add_argument("--dir", default="../Replica 8 Scene", help="Path to replica 8 scene directory (default: ../Replica 8 Scene)")
    p_data.add_argument("--scene", default=None, help="Specify a single scene to upload (e.g., office0)")

    # Command upload-spinnerf (SPIn-NeRF)
    p_spin = subparsers.add_parser("upload-spinnerf", help="Upload SPIn-NeRF dataset to modal volume")
    p_spin.add_argument("--dir", default="../SPIn-NeRF", help="Path to SPIn-NeRF directory (default: ../SPIn-NeRF)")
    p_spin.add_argument("--scene", default=None, help="Specify a single scene to upload (e.g., statue)")

    # Command upload-lerf (LERF cho InFusion)
    p_lerf = subparsers.add_parser("upload-lerf", help="Upload a LERF scene (images + transforms.json + prompts) to modal volume")
    p_lerf.add_argument("--scene", required=True, help="figurines or teatime")
    p_lerf.add_argument("--dir", default="F:/LERF Datasets", help="LERF root directory (default: F:/LERF Datasets)")
    p_lerf.add_argument("--prompts", default=None, help="SAM2 prompts json (default: outputs/lerf_prompts/<scene>.json)")
    p_lerf.add_argument("--prompts-only", action="store_true", help="Only re-upload prompts.json")

    # Command upload-outputs
    p_out =subparsers.add_parser("upload-outputs", help="Upload outputs directory to modal volume")
    p_out.add_argument("--dir", default="outputs", help="Path to outputs directory (default: outputs)")
    p_out.add_argument("--target", default="Replica", help="Target folder to upload (default: Replica)")
    p_out.add_argument("--subfolder", default=None, help="Upload a specific subfolder inside replica (e.g., sfm, bounding_box_3D)")

    # Command download-outputs
    p_down = subparsers.add_parser("download-outputs", help="Download results from modal volume to local machine")
    p_down.add_argument("--target", default="splatfill", help="Target folder on volume: splatfill or Replica (default: splatfill)")
    p_down.add_argument("--scene", default=None, help="Specify a single scene to download (e.g., 1, book, trash)")
    p_down.add_argument("--remote", default=None, help="Alternative remote path (e.g. splatfill or Replica)")
    p_down.add_argument("--local", default=None, help="Local destination path (default: Based_Model/SplatFill/output)")

    # Command clean-spinnerf
    subparsers.add_parser("clean-spinnerf", help="Clean up old SPIn-NeRF data on cloud volume")

    # Command clean-splatfill
    p_clean_sf = subparsers.add_parser("clean-splatfill", help="Clean up old splatfill output for a scene on cloud volume")
    p_clean_sf.add_argument("--scene", default="1", help="Scene to clean (default: 1)")

    # Command list-outputs
    subparsers.add_parser("list-outputs", help="List outputs files on cloud volume")

    # Command list-dataset
    subparsers.add_parser("list-dataset", help="List dataset scenes on cloud volume")

    args = parser.parse_args()

    if args.command == "clean-spinnerf":
        with app.run():
            clean_spinnerf_cloud_data.remote()
    elif args.command == "clean-splatfill":
        with app.run():
            clean_splatfill_cloud_output.remote(scene=args.scene)
    elif args.command == "upload-data":
        upload_replica_data(args.dir, args.scene)
    elif args.command == "upload-spinnerf":
        upload_spinnerf_data(args.dir, args.scene)
    elif args.command == "upload-lerf":
        upload_lerf_data(args.scene, args.dir, args.prompts, args.prompts_only)
    elif args.command == "upload-outputs":
        upload_outputs(args.dir, args.target, args.subfolder)
    elif args.command == "download-outputs":
        download_outputs(target=args.target, scene=args.scene, local_dir=args.local, remote_path=args.remote)
    elif args.command == "list-outputs":
        with app.run():
            list_outputs_remote.remote()
    elif args.command == "list-dataset":
        list_dataset_local()
    else:
        print("\n=== Modal sync and storage manager ===")
        print("1. Upload data to cloud:")
        print("   python modal_sync.py clean-spinnerf               # Clean old SPIn-NeRF data on cloud volume")
        print("   python modal_sync.py upload-spinnerf              # Upload 10 SPIn-NeRF scenes from spinnerf-dataset")
        print("   python modal_sync.py upload-spinnerf --scene 1    # Upload single scene (e.g. 1, book, trash)")
        print("   python modal_sync.py upload-data                  # Upload all 8 scenes of replica")
        print("   python modal_sync.py upload-outputs               # Upload replica directory inside outputs")
        print("   python modal_sync.py upload-outputs --subfolder sfm # Upload only sfm subfolder")
        print("\n2. Check cloud volume data:")
        print("   python modal_sync.py list-dataset                 # Check uploaded dataset scenes on cloud")
        print("   python modal_sync.py list-outputs                 # Check outputs results on cloud")
        print("   python -m modal run modal_sync.py::check_connection")
        print("\n3. Download results to local machine:")
        print("   python modal_sync.py download-outputs\n")
