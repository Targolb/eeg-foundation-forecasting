#!/usr/bin/env python
import csv, argparse, random
import pandas as pd
from pathlib import Path
from src.eegff.utils.path_utils import CHBMIT_ROOT, TUH_ROOT, KAGGLE_EEG_ROOT, OUTPUT_DIR


def check_edf(path: Path):
    try:
        import mne
        mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        return "ok", ""
    except Exception as e:
        return "bad", str(e)[:300]


def check_tfrecord(path: Path):
    try:
        import tensorflow as tf
        _ = next(iter(tf.data.TFRecordDataset([str(path)]).take(1)))
        return "ok", ""
    except Exception as e:
        return "bad", str(e)[:300]


def check_csv(path: Path):
    try:
        pd.read_csv(path, nrows=100)
        return "ok", ""
    except Exception as e:
        return "bad", str(e)[:300]


def sample_files(root: Path, exts, k: int):
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    if k <= 0 or k >= len(files): return files
    return random.sample(files, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--out", default=str(OUTPUT_DIR / "integrity" / "corruption_report.csv"))
    args = ap.parse_args()
    out = Path(args.out);
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sample_files(CHBMIT_ROOT, {".edf"}, args.sample_n):
        s, msg = check_edf(p);
        rows.append({"dataset": "chbmit", "path": str(p), "status": s, "error": msg})
    for p in sample_files(TUH_ROOT, {".edf"}, args.sample_n):
        s, msg = check_edf(p);
        rows.append({"dataset": "tuh(edf)", "path": str(p), "status": s, "error": msg})
    for p in sample_files(TUH_ROOT, {".tfrecord"}, args.sample_n):
        s, msg = check_tfrecord(p);
        rows.append({"dataset": "tuh(tfrecord)", "path": str(p), "status": s, "error": msg})
    for p in sample_files(KAGGLE_EEG_ROOT, {".csv"}, args.sample_n):
        s, msg = check_csv(p);
        rows.append({"dataset": "kaggle", "path": str(p), "status": s, "error": msg})

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "path", "status", "error"])
        w.writeheader();
        w.writerows(rows)

    bad = sum(1 for r in rows if r["status"] == "bad")
    print(f"Checked {len(rows)} files; bad: {bad}")


if __name__ == "__main__":
    main()
