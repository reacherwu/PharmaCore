#!/usr/bin/env python3
"""Download HuggingFace model files with resume support and retry."""
import os
import sys
import time
import requests
from pathlib import Path

MIRROR = "https://hf-mirror.com"

FILES = {
    "facebook/esm2_t30_150M_UR50D": [
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "model.safetensors",
    ],
}


def download_file(repo: str, filename: str, local_dir: Path, max_retries: int = 20):
    url = f"{MIRROR}/{repo}/resolve/main/{filename}"
    local_path = local_dir / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing file size for resume
    existing_size = local_path.stat().st_size if local_path.exists() else 0

    for attempt in range(max_retries):
        try:
            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                print(f"  Resuming {filename} from {existing_size / 1e6:.1f} MB (attempt {attempt + 1})")
            else:
                print(f"  Downloading {filename} (attempt {attempt + 1})")

            resp = requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)

            if resp.status_code == 416:  # Range not satisfiable = file complete
                print(f"  {filename} already complete ({existing_size / 1e6:.1f} MB)")
                return True

            if resp.status_code not in (200, 206):
                print(f"  HTTP {resp.status_code}, retrying...")
                time.sleep(5)
                continue

            total = int(resp.headers.get("content-length", 0))
            if resp.status_code == 200:
                # Server doesn't support range, start over
                existing_size = 0
                mode = "wb"
            else:
                mode = "ab"

            downloaded = 0
            start = time.time()
            with open(local_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    total_dl = existing_size + downloaded
                    if total > 0:
                        pct = total_dl / (existing_size + total) * 100
                        print(f"\r  {total_dl / 1e6:.1f}/{(existing_size + total) / 1e6:.1f} MB ({pct:.0f}%) {speed / 1e6:.2f} MB/s", end="", flush=True)
                    else:
                        print(f"\r  {total_dl / 1e6:.1f} MB {speed / 1e6:.2f} MB/s", end="", flush=True)

            existing_size += downloaded
            print(f"\n  {filename} done ({existing_size / 1e6:.1f} MB)")
            return True

        except (requests.exceptions.RequestException, ConnectionError, OSError) as e:
            print(f"\n  Error: {e}")
            existing_size = local_path.stat().st_size if local_path.exists() else 0
            wait = min(5 * (attempt + 1), 30)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  FAILED after {max_retries} attempts: {filename}")
    return False


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "facebook/esm2_t30_150M_UR50D"
    local_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/esm2_150m")

    files = FILES.get(repo, ["config.json", "tokenizer_config.json", "model.safetensors"])

    print(f"Downloading {repo} to {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        if not download_file(repo, f, local_dir):
            print(f"Failed to download {f}, aborting.")
            sys.exit(1)

    print(f"\nAll files downloaded to {local_dir}")


if __name__ == "__main__":
    main()
