import os
import shutil
from pathlib import Path

blob_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--distilbert-base-uncased-finetuned-sst-2-english" / "blobs"
snapshot_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--distilbert-base-uncased-finetuned-sst-2-english" / "snapshots" / "714eb0fa89d2f80546fda750413ed43d93601a13"

print(f"Blob dir: {blob_dir}")
print(f"Snapshot dir: {snapshot_dir}")
print(f"Blob dir exists: {blob_dir.exists()}")

# Remove and recreate snapshot dir
if snapshot_dir.exists():
    shutil.rmtree(str(snapshot_dir))
snapshot_dir.mkdir(parents=True, exist_ok=True)

# Copy all files from blobs to snapshot
files_copied = 0
for blob_file in blob_dir.iterdir():
    if blob_file.is_file():
        dst = snapshot_dir / blob_file.name
        shutil.copy2(str(blob_file), str(dst))
        files_copied += 1

print(f"Copied {files_copied} files to snapshot directory")

# Verify
files_in_snapshot = sorted([f.name for f in snapshot_dir.glob("*")])
print(f"Files in snapshot ({len(files_in_snapshot)}): {files_in_snapshot}")
