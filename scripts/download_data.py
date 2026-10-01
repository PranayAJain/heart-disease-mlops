"""Download the UCI Heart Disease dataset (Cleveland subset) into data/raw/.

Source: UCI Machine Learning Repository, dataset id 45
        https://archive.ics.uci.edu/dataset/45/heart+disease
Usage:
    python scripts/download_data.py            # download + verify
    python scripts/download_data.py --force    # re-download even if present
"""
import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
MEMBER = "processed.cleveland.data"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET = RAW_DIR / MEMBER


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and not force:
        print(f"[skip] {TARGET.relative_to(ROOT)} already exists (md5={md5(TARGET)[:10]})")
        return TARGET

    print(f"[get ] {UCI_ZIP_URL}")
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            payload = resp.read()
    except Exception as exc:
        sys.exit(
            f"[fail] could not download: {exc}\n"
            "Manual fallback: open https://archive.ics.uci.edu/dataset/45/heart+disease,\n"
            f"click 'Download', unzip, and copy '{MEMBER}' into data/raw/"
        )

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.endswith(MEMBER)]
        if not names:
            sys.exit(f"[fail] {MEMBER} not found inside the zip: {zf.namelist()}")
        TARGET.write_bytes(zf.read(names[0]))

    n_rows = sum(1 for line in TARGET.read_text().splitlines() if line.strip())
    print(f"[ok  ] wrote {TARGET.relative_to(ROOT)}  rows={n_rows}  md5={md5(TARGET)[:10]}")
    return TARGET


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download UCI heart disease data")
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    download(ap.parse_args().force)
