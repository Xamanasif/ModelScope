"""Acquire UCI dataset 350 (Default of Credit Card Clients) with provenance.

Two routes, both producing data/raw/uci_credit_default.csv + source_manifest.json:

  python scripts/download_data.py                 # official UCI API + data_url
  python scripts/download_data.py --manual PATH   # a file you downloaded yourself

Manual route: open https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
and download https://archive.ics.uci.edu/static/public/350/data.csv (the CSV the
UCI API points to; header ID,X1..X23,Y). The landing-page ZIP contains the
original .xls with a two-row header; it is NOT accepted here, because reading it
needs an extra Excel dependency and a different header convention.

No synthetic fallback exists: if neither route works, there is no dataset.
The saved bytes are exactly the bytes served by UCI, so the SHA-256 refers to
the official file. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR, DATASET_FILE, MANIFEST_FILE, sha256_file  # noqa: E402
from app.data import SOURCE_TO_SEMANTIC, read_dataset, validate_dataset  # noqa: E402

API = "https://archive.ics.uci.edu/api/dataset?id=350"
LANDING = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
CITATION = ("Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C55S3H")
LICENSE = ("Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI "
           "landing page when checked on 2026-11-18. Re-check before redistributing.")


def fetch(url: str, attempts: int = 4, timeout: int = 60) -> bytes:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "modelscope-research-prototype"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except OSError as exc:  # URLError, timeouts, DNS failures
            last = exc
            print(f"attempt {i + 1}/{attempts} failed for {url}: {exc}")
            time.sleep(2 * (i + 1))
    raise SystemExit(f"Could not download {url}: {last}. Use the --manual route (see --help).")


def check_metadata_mapping(meta: dict) -> list[str]:
    """Compare live UCI variable descriptions with the mapping hard-coded in app.data."""
    problems = []
    for var in meta.get("variables", []):
        name, desc, role = var.get("name"), var.get("description"), var.get("role")
        if name not in SOURCE_TO_SEMANTIC:
            problems.append(f"unknown source variable {name}")
        elif role == "Feature" and desc != SOURCE_TO_SEMANTIC[name]:
            problems.append(f"{name}: source says {desc!r}, code assumes {SOURCE_TO_SEMANTIC[name]!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manual", type=Path, help="register a locally downloaded data.csv instead")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = DATA_DIR / DATASET_FILE, DATA_DIR / MANIFEST_FILE
    if csv_path.exists() or manifest_path.exists():
        raise SystemExit(f"Existing dataset/manifest in {DATA_DIR}; preserve it and review before replacing.")

    metadata, problems, route, source_url = None, [], "manual", None
    if args.manual:
        if not args.manual.is_file():
            raise SystemExit(f"Manual file not found: {args.manual}")
        tmp = csv_path.with_suffix(".tmp")
        shutil.copyfile(args.manual, tmp)
        try:
            metadata = json.loads(fetch(API, attempts=1, timeout=20))["data"]
        except SystemExit:
            print("Metadata API unreachable; manifest will record that the mapping was not re-checked.")
    else:
        route = "uci_api"
        metadata = json.loads(fetch(API))["data"]
        source_url = metadata["data_url"]
        tmp = csv_path.with_suffix(".tmp")
        tmp.write_bytes(fetch(source_url))

    if metadata is not None:
        problems = check_metadata_mapping(metadata)
        if problems:
            tmp.unlink(missing_ok=True)
            raise SystemExit("Source metadata disagrees with the coded schema:\n" + "\n".join(problems))

    frame = read_dataset(tmp)
    report = validate_dataset(frame)
    if not report["ok"]:
        tmp.unlink(missing_ok=True)
        failed = [c for c in report["checks"] if c["status"] == "fail"]
        raise SystemExit(f"Validation failed; file not registered: {json.dumps(failed, indent=1)}")

    manifest = {
        "uci_id": 350,
        "name": "Default of Credit Card Clients",
        "landing_page": LANDING,
        "acquisition_route": route,
        "source_url": source_url,
        "manual_source_path": str(args.manual) if args.manual else None,
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(tmp),
        "bytes": tmp.stat().st_size,
        "rows": report["rows"],
        "columns_in_file": list(SOURCE_TO_SEMANTIC),
        "citation": CITATION,
        "intro_paper_doi": "10.1016/j.eswa.2007.12.020",
        "license": LICENSE,
        "metadata_mapping_checked": metadata is not None,
        "target_semantics": "Y = default payment next month; source: 'default payment (Yes = 1, No = 0)'",
        "source_metadata_last_updated": (metadata or {}).get("last_updated"),
        "validation_summary": report["summary"],
        "redistribution": "Data files are git-ignored. Cite the source; check license before sharing.",
    }
    tmp.replace(csv_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {report['rows']} rows to {csv_path}")
    print(f"SHA-256 {manifest['sha256']}")
    print(f"Validation summary: {report['summary']}")


if __name__ == "__main__":
    main()
