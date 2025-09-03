# src/eegff/overview/02_session_overview.py
# !/usr/bin/env python
import os
import re
import math
import csv
from pathlib import Path

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


# ---------- Common helpers ----------
def edf_meta(edf_path: Path):
    """Return (duration_hours, n_channels, sfreq) for an EDF without preloading."""
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"]) if raw.info.get("sfreq") else float("nan")
    n_times = int(raw.n_times) if raw.n_times is not None else 0
    secs = n_times / sfreq if (sfreq and sfreq > 0) else float("nan")
    hours = secs / 3600.0 if math.isfinite(secs) else float("nan")
    return hours, len(raw.ch_names), sfreq


def is_edf(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".edf") or n.endswith(".edf.gz")


# ---------- CHB-MIT ----------
def chb_subject_from_path(p: Path) -> str:
    # typical: /.../chb01/<file>.edf
    for part in p.parts:
        if re.fullmatch(r"chb\d{2}", part.lower()):
            return part
    return ""


def chb_parse_seizures_for_file(edf_path: Path):
    """
    Look for seizure annotations for a specific EDF:
      1) <edf_stem>.seizures  (same dir)
         - lines like: 'seizure n start time : <sec> end time : <sec>'
      2) *summary*.txt        (same dir or subject dir)
         - try to count blocks tied to this file.
    Returns: (n_seizures, has_annotation(bool), source_str)
    """
    parent = edf_path.parent
    stem = edf_path.stem  # handles .edf.gz -> 'xxx.edf' as stem; handle both
    # normalize to base without trailing .edf if present in stem
    if stem.lower().endswith(".edf"):
        base = stem[:-4]
    else:
        base = stem

    # 1) file-specific .seizures
    seizures_file = parent / f"{base}.seizures"
    if seizures_file.exists():
        try:
            txt = seizures_file.read_text(errors="ignore")
            # count lines that look like 'seizure ... start ... end ...'
            hits = re.findall(r"start\s*time\s*:\s*\d+.*?end\s*time\s*:\s*\d+", txt, flags=re.I)
            return len(hits), True, str(seizures_file.name)
        except Exception:
            pass

    # 2) summary files (in parent and one level up)
    candidates = list(parent.glob("*summary*.txt"))
    subj_dir = parent.parent
    candidates += list(subj_dir.glob("*summary*.txt")) if subj_dir.exists() else []
    # Heuristic: within the summary, sections per file start with the filename
    for summ in candidates:
        try:
            txt = summ.read_text(errors="ignore")
            # Find the section for this file (base or base.edf)
            # Greedy but local: get lines containing the filename and subsequent seizure lines
            # Count "Seizure <n>" occurrences under that file section.
            # As a fallback, count "Number of Seizures in File <file> : N"
            # 2a) Exact per-file total:
            pat_total = re.compile(rf"Number of Seizures in File\s+{re.escape(base)}(?:\.edf)?\s*:\s*(\d+)", re.I)
            m = pat_total.search(txt)
            if m:
                return int(m.group(1)), True, summ.name
            # 2b) Count explicit seizure entries that include the filename on the same or nearby lines
            # build a small window around occurrences of the filename and count 'Seizure ' tokens in next lines
            count = 0
            for hit in re.finditer(re.escape(base), txt, flags=re.I):
                window = txt[hit.start(): hit.start() + 2000]  # next ~2KB
                count += len(re.findall(r"\bSeizure\s+\d+\b", window, flags=re.I))
            if count > 0:
                return count, True, summ.name
        except Exception:
            continue

    return 0, False, ""


def chb_sessions():
    rows = []
    if not CHBMIT_ROOT.exists():
        return rows
    edfs = [p for p in CHBMIT_ROOT.rglob("*") if p.is_file() and is_edf(p)]
    for f in sorted(edfs):
        try:
            dur_h, n_ch, sfreq = edf_meta(f)
        except Exception:
            dur_h, n_ch, sfreq = float("nan"), "", ""
        sid = chb_subject_from_path(f)
        # CHB "session": we’ll use the parent folder name (often all files in subject root)
        session_id = f.parent.name
        n_seiz, has_ann, source = chb_parse_seizures_for_file(f)
        rows.append({
            "dataset": "chbmit",
            "subject_id": sid,
            "session_id": session_id,
            "file_relpath": str(f.relative_to(CHBMIT_ROOT)),
            "duration_hours": round(dur_h, 6) if math.isfinite(dur_h) else "",
            "n_channels": n_ch,
            "sfreq": round(float(sfreq), 3) if sfreq not in ("", None) and math.isfinite(float(sfreq)) else "",
            "n_seizures": n_seiz if n_seiz else "",
            "has_annotation": int(has_ann),
            "annotation_source": source,
            "cohort": "",
        })
    return rows


# ---------- TUH (v2.0.1) ----------
def tuh_subject_from_relpath(root: Path, p: Path) -> str:
    rel = p.relative_to(root)
    parts = rel.parts
    return parts[1] if len(parts) >= 2 else ""


def tuh_session_from_relpath(root: Path, p: Path) -> str:
    rel = p.relative_to(root)
    parts = rel.parts
    # typical: cohort/subject/s001_2012/montage/file.edf
    return parts[2] if len(parts) >= 3 else ""


def tuh_cohort_from_relpath(root: Path, p: Path) -> str:
    rel = p.relative_to(root)
    return rel.parts[0] if len(rel.parts) >= 1 else ""


def tuh_sessions():
    rows = []
    if not TUH_ROOT.exists():
        return rows
    for f in TUH_ROOT.rglob("*"):
        if not (f.is_file() and is_edf(f)):
            continue
        try:
            dur_h, n_ch, sfreq = edf_meta(f)
        except Exception:
            dur_h, n_ch, sfreq = float("nan"), "", ""
        sid = tuh_subject_from_relpath(TUH_ROOT, f)
        session_id = tuh_session_from_relpath(TUH_ROOT, f)
        cohort = tuh_cohort_from_relpath(TUH_ROOT, f)
        rows.append({
            "dataset": "tuh",
            "subject_id": sid,
            "session_id": session_id,
            "file_relpath": str(f.relative_to(TUH_ROOT)),
            "duration_hours": round(dur_h, 6) if math.isfinite(dur_h) else "",
            "n_channels": n_ch,
            "sfreq": round(float(sfreq), 3) if sfreq not in ("", None) and math.isfinite(float(sfreq)) else "",
            "n_seizures": "",  # Not parsed here
            "has_annotation": 0,  # Annotations are external in TUH baseline
            "annotation_source": "",
            "cohort": cohort,  # 00_epilepsy / 01_no_epilepsy
        })
    return rows


# ---------- Kaggle CSV ----------
def kaggle_session_meta(csv_path: Path):
    """
    Estimate (duration_hours, n_channels, sfreq) for a CSV:
      - If 'sfreq'/'sampling_rate' present -> sfreq, duration = nrows/sfreq
      - Else if 'time' column present -> duration from max-min, sfreq unknown
      - n_channels = number of columns minus clear non-signal columns
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return "", "", ""
    # sampling rate
    fs = None
    for k in ["sfreq", "sampling_rate", "fs", "Fs"]:
        if k in df.columns:
            try:
                fs = float(df[k].dropna().iloc[0])
                break
            except Exception:
                pass
    # duration
    if fs and fs > 0:
        dur_h = len(df) / fs / 3600.0
    elif "time" in df.columns:
        try:
            t = df["time"].astype(float)
            dur_h = float(t.max() - t.min()) / 3600.0
        except Exception:
            dur_h = ""
    else:
        dur_h = ""
    # channel count: count numeric columns excluding obvious non-signal
    nonsignal = {"time", "label", "seizure", "subject", "id", "sfreq", "sampling_rate", "fs", "Fs"}
    num_cols = 0
    for c in df.columns:
        if c in nonsignal:
            continue
        try:
            pd.to_numeric(df[c].head(32), errors="raise")
            num_cols += 1
        except Exception:
            continue
    return (round(dur_h, 6) if isinstance(dur_h, float) and math.isfinite(dur_h) else "",
            num_cols if num_cols > 0 else "",
            round(fs, 3) if fs else "")


def kaggle_subject_from_csv(path: Path):
    m = re.match(r"(s\d+)", path.stem.lower())
    return m.group(1) if m else path.stem


def kaggle_sessions():
    rows = []
    if not KAGGLE_EEG_ROOT.exists():
        return rows
    csvs = [p for p in KAGGLE_EEG_ROOT.rglob("*.csv")]
    for f in sorted(csvs):
        dur_h, n_ch, sfreq = kaggle_session_meta(f)
        rows.append({
            "dataset": "kaggle",
            "subject_id": kaggle_subject_from_csv(f),
            "session_id": "",  # unknown
            "file_relpath": str(f.relative_to(KAGGLE_EEG_ROOT)),
            "duration_hours": dur_h,
            "n_channels": n_ch,
            "sfreq": sfreq,
            "n_seizures": "",
            "has_annotation": 0,
            "annotation_source": "",
            "cohort": "",
        })
    return rows


# ---------- Main ----------
def main():
    rows = []
    rows += chb_sessions()
    rows += tuh_sessions()
    rows += kaggle_sessions()

    fieldnames = [
        "dataset", "subject_id", "session_id", "file_relpath",
        "duration_hours", "n_channels", "sfreq",
        "n_seizures", "has_annotation", "annotation_source", "cohort"
    ]
    # Ensure keys present
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, "")

    out_csv = OUTDIR / "session_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {out_csv} with {len(rows)} sessions.")


if __name__ == "__main__":
    main()
