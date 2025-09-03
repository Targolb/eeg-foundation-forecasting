# src/eegff/overview/01_subject_overview.py
# !/usr/bin/env python
import os
import re
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter

# ---- Optional dotenv (safe if not installed) ----
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ---- Config via env (set in Docker -e ... or .env) ----
CHBMIT_ROOT = Path(os.getenv("CHBMIT_ROOT", "")).expanduser().resolve()
TUH_ROOT = Path(os.getenv("TUH_ROOT", "")).expanduser().resolve()
KAGGLE_EEG_ROOT = Path(os.getenv("KAGGLE_EEG_ROOT", "")).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).expanduser().resolve()
OUTDIR = OUTPUT_DIR / "overview"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---- Lazy import MNE only when needed ----
def edf_duration_seconds(edf_path: Path):
    """
    Return duration (seconds), sfreq, n_channels for an EDF without loading data.
    """
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"]) if raw.info.get("sfreq") else float("nan")
    n_times = int(raw.n_times) if raw.n_times is not None else 0
    secs = n_times / sfreq if sfreq and sfreq > 0 else float("nan")
    return secs, sfreq, len(raw.ch_names)


# ----------------------------
# CHB-MIT helpers
# ----------------------------
def chb_subject_id_from_dir(p: Path) -> str:
    # typical dirs: chb01, chb02, ...
    return p.name


def chb_find_edfs(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".edf",)]


def chb_parse_summary_texts(subj_dir: Path):
    """
    Parse chbXX-summary.txt if present to count seizures per subject.
    Falls back to 0 if not found.
    """
    seizure_count = 0
    for txt in subj_dir.glob("*summary*.txt"):
        try:
            content = txt.read_text(errors="ignore")
            # Count occurrences like "Seizure" lines or "Number of Seizures in File ..."
            # Heuristic: count "Seizure " occurrences that look like event lines.
            # Also parse patterns: 'Number of Seizures in File ...: N'
            seizure_count += len(re.findall(r"\bSeizure\s+\d+\b", content, flags=re.I))
            for m in re.finditer(r"Number of Seizures.*?:\s*(\d+)", content, flags=re.I):
                seizure_count += int(m.group(1))
        except Exception:
            pass
    return seizure_count


def chb_overview():
    rows = []
    if not CHBMIT_ROOT.exists():
        return rows

    # group by subject folder
    subj_dirs = [d for d in CHBMIT_ROOT.iterdir() if d.is_dir() and d.name.lower().startswith("chb")]
    for sdir in sorted(subj_dirs):
        sid = chb_subject_id_from_dir(sdir)
        edfs = [p for p in sdir.glob("*.edf")]
        # if EDFs also appear deeper, include them too:
        if not edfs:
            edfs = chb_find_edfs(sdir)

        n_recs = len(edfs)
        n_sessions = len({p.parent for p in edfs})  # rough session proxy = parent folders
        total_secs = 0.0
        for f in edfs:
            try:
                secs, _, _ = edf_duration_seconds(f)
                if math.isfinite(secs):
                    total_secs += secs
            except Exception:
                # If one file fails header read, just skip duration
                pass

        # Try seizure counts via summary files (heuristic)
        n_seiz = chb_parse_summary_texts(sdir)

        rows.append({
            "dataset": "chbmit",
            "subject_id": sid,
            "n_sessions": n_sessions,
            "n_recordings": n_recs,
            "total_hours": round(total_secs / 3600.0, 3),
            "n_seizures": n_seiz if n_seiz > 0 else "",  # blank if unknown/zero
        })
    return rows


# ----------------------------
# TUH (v2.0.1) helpers
# ----------------------------
# Expected structure (examples):
#  TUH_ROOT/00_epilepsy/<subject>/s001_2012/02_tcp_le/<file>.edf
#  TUH_ROOT/01_no_epilepsy/<subject>/s001_2011/01_tcp_ar/<file>.edf
TUH_COHORTS = {"00_epilepsy", "01_no_epilepsy"}


def tuh_is_edf(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".edf") or n.endswith(".edf.gz")


def tuh_subject_from_relpath(rel: Path):
    # rel parts: cohort / subject / session / montage / file
    parts = rel.parts
    subj = parts[1] if len(parts) >= 2 else ""
    return subj


def tuh_session_from_relpath(rel: Path):
    # e.g., s001_2012 (if present)
    parts = rel.parts
    return parts[2] if len(parts) >= 3 else ""


def tuh_overview():
    rows = []
    if not TUH_ROOT.exists():
        return rows

    # collect per-subject stats
    per_subj_recs = defaultdict(list)  # subject -> list of EDF files
    per_subj_sessions = defaultdict(set)
    per_subj_secs = defaultdict(float)
    per_subj_cohort = {}  # last seen cohort label per subject

    for cohort_dir in TUH_COHORTS:
        base = TUH_ROOT / cohort_dir
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.is_file() and tuh_is_edf(f):
                rel = f.relative_to(TUH_ROOT)
                sid = tuh_subject_from_relpath(rel)
                sess = tuh_session_from_relpath(rel)
                per_subj_recs[sid].append(f)
                if sess:
                    per_subj_sessions[sid].add(sess)
                per_subj_cohort[sid] = cohort_dir

    for sid, files in per_subj_recs.items():
        total_secs = 0.0
        for f in files:
            try:
                secs, _, _ = edf_duration_seconds(f)
                if math.isfinite(secs):
                    total_secs += secs
            except Exception:
                pass

        rows.append({
            "dataset": "tuh",
            "subject_id": sid,
            "n_sessions": len(per_subj_sessions[sid]) if per_subj_sessions[sid] else "",
            "n_recordings": len(files),
            "total_hours": round(total_secs / 3600.0, 3),
            # TUH seizure *event* counts require separate annotations; for now:
            "n_seizures": "",  # unknown here
            "cohort": per_subj_cohort.get(sid, ""),  # 00_epilepsy / 01_no_epilepsy
        })
    return rows


# ----------------------------
# Kaggle EEG helpers (generic CSVs)
# ----------------------------
def kaggle_subject_from_csv(path: Path):
    # Heuristic: s01.csv or similar → s01; otherwise use stem.
    m = re.match(r"(s\d+)", path.stem.lower())
    return m.group(1) if m else path.stem


def kaggle_duration_seconds(csv_path: Path):
    """
    Try to estimate duration from CSV:
    - If column named 'sfreq'/'sampling_rate' exists → rows / sfreq
    - Else if 'time' column exists → infer from min/max
    - Else return NaN
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        for k in ["sfreq", "sampling_rate", "fs", "Fs"]:
            if k in df.columns:
                try:
                    fs = float(df[k].dropna().iloc[0])
                    if fs > 0:
                        return len(df) / fs
                except Exception:
                    pass
        if "time" in df.columns:
            t = df["time"].astype(float)
            if len(t) > 1:
                return float(t.max() - t.min())
    except Exception:
        return float("nan")
    return float("nan")


def kaggle_overview():
    rows = []
    if not KAGGLE_EEG_ROOT.exists():
        return rows
    csvs = [p for p in KAGGLE_EEG_ROOT.rglob("*.csv")]
    per_subj_files = defaultdict(list)
    per_subj_secs = defaultdict(float)

    for f in csvs:
        sid = kaggle_subject_from_csv(f)
        per_subj_files[sid].append(f)
        secs = kaggle_duration_seconds(f)
        if math.isfinite(secs):
            per_subj_secs[sid] += secs

    for sid, files in per_subj_files.items():
        rows.append({
            "dataset": "kaggle",
            "subject_id": sid,
            "n_sessions": "",  # unknown/not applicable
            "n_recordings": len(files),
            "total_hours": round(per_subj_secs[sid] / 3600.0, 3) if per_subj_secs[sid] else "",
            "n_seizures": "",  # no seizure annotations here
        })
    return rows


# ----------------------------
# Main
# ----------------------------
def main():
    rows = []
    rows += chb_overview()
    rows += tuh_overview()
    rows += kaggle_overview()

    # Standardize columns & write CSV
    fieldnames = ["dataset", "subject_id", "n_sessions", "n_recordings", "total_hours", "n_seizures", "cohort"]
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, "")

    out_csv = OUTDIR / "subject_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["dataset"], x["subject_id"])):
            w.writerow(r)

    print(f"Wrote {out_csv} with {len(rows)} subjects.")


if __name__ == "__main__":
    main()
