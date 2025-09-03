#!/usr/bin/env python
import csv, argparse, statistics, random
from pathlib import Path
from src.eegff.utils.path_utils import CHBMIT_ROOT, TUH_ROOT, KAGGLE_EEG_ROOT, OUTPUT_DIR

def edf_rate(path: Path):
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    return float(raw.info["sfreq"])

def csv_rate(path: Path):
    # Kaggle CSVs vary; many have a header with sampling rate OR infer from 'time' delta if present.
    # Try a few heuristics; fallback to None.
    import pandas as pd
    try:
        df = pd.read_csv(path, nrows=400)
        for k in ["sfreq","sampling_rate","fs","Fs"]:
            if k in df.columns:
                v = df[k].dropna().iloc[0]
                try: return float(v)
                except: pass
        if "time" in df.columns:
            # assume seconds; estimate from first 200 deltas
            t = df["time"].astype(float).values
            if len(t) > 5:
                dt = abs(t[1:200]-t[:199])
                med = float(sorted(dt)[len(dt)//2])
                if med > 0: return 1.0/med
    except Exception:
        pass
    return None

def sample(root: Path, exts, n):
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return files if n<=0 or n>=len(files) else random.sample(files, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=80)
    ap.add_argument("--outdir", default=str(OUTPUT_DIR / "integrity"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sample(CHBMIT_ROOT, {".edf"}, args.sample_n):
        rows.append({"dataset":"chbmit","file":str(p), "sfreq":edf_rate(p)})
    for p in sample(TUH_ROOT, {".edf"}, args.sample_n):
        rows.append({"dataset":"tuh(edf)","file":str(p), "sfreq":edf_rate(p)})
    for p in sample(KAGGLE_EEG_ROOT, {".csv"}, args.sample_n):
        rows.append({"dataset":"kaggle","file":str(p), "sfreq":csv_rate(p)})

    # write per-file
    with open(outdir / "sfreq_per_file.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","file","sfreq"])
        w.writeheader(); w.writerows(rows)

    # summarize
    def summarize(ds):
        vals = [r["sfreq"] for r in rows if r["dataset"]==ds and r["sfreq"]]
        if not vals: return {"count":0}
        return {
            "count": len(vals),
            "mean": round(sum(vals)/len(vals),2),
            "median": round(statistics.median(vals),2),
            "min": round(min(vals),2),
            "max": round(max(vals),2),
        }

    summary = []
    for ds in ["chbmit","tuh(edf)","kaggle"]:
        s = summarize(ds); s["dataset"]=ds; summary.append(s)

    with open(outdir / "sfreq_summary.csv", "w", newline="") as f:
        keys = ["dataset","count","mean","median","min","max"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for s in summary: w.writerow({k:s.get(k,"") for k in keys})

    print("Sampling-rate summary written.")

if __name__ == "__main__":
    main()
