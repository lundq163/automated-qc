#!/usr/bin/env python3
"""
copy_from_csv.py

Copy (or symlink) files listed in a CSV from a source directory to a destination
directory. Designed for BIDS-ish filenames so far used in automated-QC, like this:

subject_id,session_id,run_id,suffix,scan,QU_motion
100079,V02,1,T2w,sub-100079_ses-V02_run-1_T2w.nii.gz,3.0

Usage examples:
  # Basic: find files by scan filename anywhere under src and copy them to dst
  python copy_from_csv.py --csv needs.csv --src /data/source --dst /data/selected

  # Use BIDS layout (sub-<id>/ses-<id>/<modality>/<scan>) for faster lookups
  python copy_from_csv.py --csv needs.csv --src /bids --dst /pick --layout bids

  # Only take scans with QU_motion <= 1.5 (if that column exists)
  python copy_from_csv.py --csv needs.csv --src /bids --dst /pick --max-motion 1.5

  # Dry run first; then symlink instead of copying
  python copy_from_csv.py --csv needs.csv --src /bids --dst /pick --dry-run
  python copy_from_csv.py --csv needs.csv --src /bids --dst /pick --symlink
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import shutil
import concurrent.futures as futures

# Map common MRI suffixes to BIDS modality directories
MODALITY_MAP = {
    "T1w": "anat",
    "T2w": "anat",
    # "FLAIR": "anat",
    # "PD": "anat",
    # "bold": "func",
    # "sbref": "func",
    # "dwi": "dwi",
    # "ADC": "dwi",
    # "phasediff": "fmap",
    # "magnitude1": "fmap",
    # "magnitude2": "fmap",
    # "phase1": "fmap",
    # "phase2": "fmap",
    # Feel free to extend as needed
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Copy/symlink scan files listed in a CSV from src to dst."
    )
    p.add_argument("--csv", required=True, help="Input CSV path.")
    p.add_argument("--src", required=True, help="Source root directory.")
    p.add_argument("--dst", required=True, help="Destination root directory.")
    p.add_argument("--layout", choices=["auto", "bids", "search"], default="auto",
                   help="How to locate files: 'bids' (fast, structured), "
                        "'search' (rglob by filename), or 'auto' (try bids then search).")
    p.add_argument("--filename-column", default="scan",
                   help="CSV column containing the filename to copy (default: scan).")
    p.add_argument("--subject-column", default="subject_id",
                   help="CSV column for subject (default: subject_id).")
    p.add_argument("--session-column", default="session_id",
                   help="CSV column for session (default: session_id).")
    p.add_argument("--suffix-column", default="suffix",
                   help="CSV column for suffix (default: suffix).")
    p.add_argument("--max-motion", type=float, default=None,
                   help="Optional: only copy rows where QU_motion <= this value.")
    p.add_argument("--motion-column", default="QU_motion",
                   help="CSV column containing motion metric (default: QU_motion).")
    p.add_argument("--dry-run", action="store_true",
                   help="List actions without copying.")
    p.add_argument("--symlink", action="store_true",
                   help="Create symlinks instead of copying bytes.")
    p.add_argument("--jobs", type=int, default=8,
                   help="Parallel copy/symlink workers (default: 8).")
    p.add_argument("--preserve-tree", action="store_true",
                   help="Mirror BIDS sub/ses/modality folder structure under dst "
                        "(default: place all files at dst root unless bids layout used).")
    return p.parse_args()

def read_rows(csv_path: Path):
    with csv_path.open(newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("No rows found in CSV.", file=sys.stderr)
    return rows

def motion_ok(row: dict, motion_col: str, max_motion: float | None):
    if max_motion is None:
        return True
    val = row.get(motion_col, "")
    try:
        return float(val) <= max_motion
    except Exception:
        # If motion missing or unparsable, skip row when filtering
        return False

def guess_modality(suffix: str | None) -> str | None:
    if not suffix:
        return None
    return MODALITY_MAP.get(suffix, None)

def bids_candidate_paths(src_root: Path, row: dict,
                         fname_col: str, subj_col: str, ses_col: str, suf_col: str):
    """Yield likely BIDS paths for the file."""
    fname = row.get(fname_col, "").strip()
    if not fname:
        return
    sub = row.get(subj_col, "").strip()
    ses = row.get(ses_col, "").strip()
    suf = row.get(suf_col, "").strip()

    sub_tag = f"sub-{sub}" if not str(sub).startswith("sub-") and sub else sub
    ses_tag = f"ses-{ses}" if ses and not str(ses).startswith("ses-") else ses

    # Prefer modality from suffix map; also try common fallbacks
    primary_mod = guess_modality(suf)
    modality_order = [m for m in [primary_mod, "anat", "func", "dwi", "fmap"] if m]

    for mod in modality_order:
        parts = [src_root]
        if sub_tag:
            parts.append(sub_tag)
        if ses_tag:
            parts.append(ses_tag)
        parts.append(mod)
        yield Path(*parts) / fname

    # As a final structured guess, also try sub-only paths (no session)
    if sub_tag:
        for mod in modality_order:
            yield src_root / sub_tag / mod / fname

def search_by_filename(src_root: Path, fname: str):
    # rglob exact filename anywhere under src_root
    # Using rglob with '**/fname' risks performance; but acceptable for moderate trees.
    return list(src_root.rglob(fname))

def choose_locator(layout: str):
    if layout == "bids":
        return ("bids",)
    if layout == "search":
        return ("search",)
    # auto: try bids first, then search
    return ("bids", "search")

def make_dst_path(dst_root: Path, row: dict, fname: str, preserve_tree: bool,
                  subj_col: str, ses_col: str, suf_col: str):
    if preserve_tree:
        sub = row.get(subj_col, "").strip()
        ses = row.get(ses_col, "").strip()
        suf = row.get(suf_col, "").strip()
        sub_tag = f"sub-{sub}" if sub and not str(sub).startswith("sub-") else sub
        ses_tag = f"ses-{ses}" if ses and not str(ses).startswith("ses-") else ses
        mod = guess_modality(suf) or "unknown"
        parts = [dst_root]
        if sub_tag:
            parts.append(sub_tag)
        if ses_tag:
            parts.append(ses_tag)
        parts.append(mod)
        return Path(*parts) / fname
    else:
        return dst_root / fname

def copy_or_link(src: Path, dst: Path, symlink: bool, dry_run: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return True, "DRY-RUN"
    try:
        if symlink:
            # Replace existing links/files
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        return True, "OK"
    except Exception as e:
        return False, f"ERROR: {e}"

def process_row(row: dict, cfg) -> tuple[str, str, str]:
    """
    Returns (status, src_path_str, dst_path_str)
    status: 'COPIED' | 'LINKED' | 'SKIPPED' | 'MISSING' | 'FILTERED'
    """
    # motion filter
    if not motion_ok(row, cfg.motion_column, cfg.max_motion):
        return ("FILTERED", "", "")

    fname = row.get(cfg.filename_column, "").strip()
    if not fname:
        return ("SKIPPED", "", "")

    src_root = Path(cfg.src)
    dst_root = Path(cfg.dst)

    src_path: Path | None = None
    for mode in choose_locator(cfg.layout):
        if mode == "bids":
            for cand in bids_candidate_paths(src_root, row,
                                             cfg.filename_column, cfg.subject_column,
                                             cfg.session_column, cfg.suffix_column):
                if cand.exists():
                    src_path = cand
                    break
            if src_path:
                break
        elif mode == "search":
            hits = search_by_filename(src_root, fname)
            if hits:
                # If multiple, prefer shortest path (likely more canonical)
                src_path = sorted(hits, key=lambda p: len(str(p)))[0]
                break

    if not src_path:
        return ("MISSING", "", "")

    dst_path = make_dst_path(dst_root, row, fname, cfg.preserve_tree,
                             cfg.subject_column, cfg.session_column, cfg.suffix_column)

    ok, msg = copy_or_link(src_path, dst_path, cfg.symlink, cfg.dry_run)
    if not ok:
        return ("SKIPPED", str(src_path), str(dst_path))

    if cfg.dry_run:
        return ("SKIPPED", str(src_path), str(dst_path))
    return ("LINKED" if cfg.symlink else "COPIED", str(src_path), str(dst_path))

def write_manifest(rows: list[tuple[str, str, str]], dst_root: Path):
    copied = [r for r in rows if r[0] in ("COPIED", "LINKED")]
    missing = [r for r in rows if r[0] == "MISSING"]
    filtered = [r for r in rows if r[0] == "FILTERED"]

    def _write(name, data, headers):
        path = dst_root / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(data)
        return path

    manifests = []
    if copied:
        manifests.append(_write("copied_manifest.csv", copied,
                                ["status", "src_path", "dst_path"]))
    if missing:
        manifests.append(_write("missing_manifest.csv", missing,
                                ["status", "src_path", "dst_path"]))
    if filtered:
        manifests.append(_write("filtered_manifest.csv", filtered,
                                ["status", "src_path", "dst_path"]))

    return manifests

def main():
    cfg = parse_args()
    csv_path = Path(cfg.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)
    if not Path(cfg.src).exists():
        print(f"Source root not found: {cfg.src}", file=sys.stderr)
        sys.exit(2)
    Path(cfg.dst).mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    if not rows:
        sys.exit(1)

    to_process = []
    for r in rows:
        to_process.append(r)

    results: list[tuple[str, str, str]] = []
    with futures.ThreadPoolExecutor(max_workers=max(cfg.jobs, 1)) as ex:
        for status, srcp, dstp in ex.map(lambda r: process_row(r, cfg), to_process):
            results.append((status, srcp, dstp))

    # Tally
    counts = {}
    for s, _, _ in results:
        counts[s] = counts.get(s, 0) + 1

    manifests = write_manifest(results, Path(cfg.dst))

    # Summary
    print("\n=== Summary ===")
    for k in sorted(counts):
        print(f"{k:9s}: {counts[k]}")
    if manifests:
        print("\nManifests written to:")
        for m in manifests:
            print(f" - {m}")

if __name__ == "__main__":
    main()
