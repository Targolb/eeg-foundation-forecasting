# src/eegff/labels/03_label_distribution.py
# !/usr/bin/env python
import os
import re
import csv
import math
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

# Optional dotenv
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# --------- Env / Paths ---------
CHBMIT_ROOT = Path(os.getenv("CHBMIT_ROOT", "")).expanduser().resolve()
TUH_ROOT = Path(os.getenv("TUH_ROOT", "")).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).expanduser().resolve()

OUTDIR = OUTPUT_DIR / "labels"
PLOTDIR = OUTDIR / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)


# --------- Common helpers ---------
def is_edf(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".edf") or n.endswith(".edf.gz")


def edf_duration_seconds(edf_path: Path) -> float:
    """Get total duration (seconds) from EDF header (no preload)."""
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"]) if raw.info.get("sfreq") else float("nan")
    n_times = int(raw.n_times) if raw.n_times is not None else 0
    return float(n_times / sfreq) if (sfreq and sfreq > 0) else float("nan")


def clip_segments_to_total(segments: List[Tuple[float, float]], total: float) -> List[Tuple[float, float]]:
    """Ensure segments are inside [0, total] and start<end; merge overlaps."""
    cleaned = []
    for s, e in segments:
        if not (math.isfinite(s) and math.isfinite(e)):
            continue
        a, b = max(0.0, min(s, e)), min(total, max(s, e))
        if b > a:
            cleaned.append((a, b))
    if not cleaned:
        return []
    # merge overlaps
    cleaned.sort()
    merged = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def sum_durations(segments: List[Tuple[float, float]]) -> float:
    return sum((e - s) for s, e in segments)


# --------- CHB-MIT parsing ---------
def chb_subject_from_path(p: Path) -> str:
    for part in p.parts:
        if re.fullmatch(r"chb\d{2}", part.lower()):
            return part
    return ""


def chb_parse_seizures_for_file(edf_path: Path) -> List[Tuple[float, float]]:
    """
    Return seizure [start, end] (seconds) for this EDF by:
      1) <stem>.seizures with 'start time : <sec> end time : <sec>' lines
      2) *summary*.txt sections mentioning this file with 'Seizure k' or 'start/end' lines
    """
    segments = []

    parent = edf_path.parent
    stem = edf_path.stem
    base = stem[:-4] if stem.lower().endswith(".edf") else stem

    # 1) file-specific .seizures
    seizures_file = parent / f"{base}.seizures"
    if seizures_file.exists():
        try:
            txt = seizures_file.read_text(errors="ignore")
            for m in re.finditer(r"start\s*time\s*:\s*([0-9.]+)\s*.*?end\s*time\s*:\s*([0-9.]+)", txt,
                                 flags=re.I | re.S):
                s = float(m.group(1));
                e = float(m.group(2))
                segments.append((s, e))
        except Exception:
            pass

    # 2) summary files (parent and one level up)
    candidates = list(parent.glob("*summary*.txt"))
    subj_dir = parent.parent
    candidates += list(subj_dir.glob("*summary*.txt")) if subj_dir.exists() else []
    # extract per-file window and parse
    fname_regex = re.compile(rf"{re.escape(base)}(?:\.edf)?", re.I)
    for summ in candidates:
        try:
            txt = summ.read_text(errors="ignore")
        except Exception:
            continue
        # iterate all occurrences of the filename and parse ~2k chars window after
        for hit in fname_regex.finditer(txt):
            window = txt[hit.start(): hit.start() + 2000]
            # explicit "start time/end time" pairs
            for m in re.finditer(r"start\s*time\s*:\s*([0-9.]+)\s*.*?end\s*time\s*:\s*([0-9.]+)", window,
                                 flags=re.I | re.S):
                s = float(m.group(1));
                e = float(m.group(2))
                segments.append((s, e))
            # sometimes they write "Seizure k from X to Y seconds"
            for m in re.finditer(r"Seizure\s+\d+.*?(?:from|start)\s*([0-9.]+)\s*(?:to|end)\s*([0-9.]+)\s*seconds",
                                 window, flags=re.I | re.S):
                s = float(m.group(1));
                e = float(m.group(2))
                segments.append((s, e))

    # dedupe/merge after clipping later
    return segments


def chb_file_rows() -> List[Dict]:
    rows = []
    if not CHBMIT_ROOT.exists():
        return rows
    for f in CHBMIT_ROOT.rglob("*"):
        if f.is_file() and is_edf(f):
            sid = chb_subject_from_path(f)
            total = edf_duration_seconds(f)
            segs = chb_parse_seizures_for_file(f)
            segs = clip_segments_to_total(segs, total if math.isfinite(total) else float("inf"))
            seiz = sum_durations(segs)
            non = max(0.0, (total - seiz)) if math.isfinite(total) else ""
            rows.append({
                "dataset": "chbmit",
                "subject_id": sid,
                "file_relpath": str(f.relative_to(CHBMIT_ROOT)),
                "total_seconds": round(total, 6) if math.isfinite(total) else "",
                "seizure_seconds": round(seiz, 6),
                "nonseizure_seconds": round(non, 6) if non != "" else "",
                "n_seizure_events": len(segs),
            })
    return rows


# --------- TUH parsing (best-effort) ---------
# We try common sidecar formats: .tse, .tse_bi, .csv with columns like start,end,label or start,duration,label
def tuh_subject_from_rel(root: Path, p: Path) -> str:
    rel = p.relative_to(root)
    return rel.parts[1] if len(rel.parts) >= 2 else ""


def tuh_find_sidecar(edf_path: Path) -> Path | None:
    base = edf_path.with_suffix("")  # remove .edf or .gz
    candidates = [
        edf_path.with_suffix(".tse_bi"),
        edf_path.with_suffix(".tse"),
        edf_path.with_suffix(".csv"),
        Path(str(base).replace(".edf", "") + ".tse_bi"),
        Path(str(base).replace(".edf", "") + ".tse"),
        Path(str(base).replace(".edf", "") + ".csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def tuh_parse_sidecar(sidecar: Path) -> List[Tuple[float, float, str]]:
    """
    Parse a sidecar file into list of (start, end, label).
    Accepts:
      - Space/tab delimited lines: start end label
      - Or: start duration label  (we'll convert)
      - CSV with columns start,end,label OR start,duration,label
    Returns only rows with a label containing 'seiz' (case-insensitive).
    """
    rows = []
    txt = None
    try:
        txt = sidecar.read_text(errors="ignore")
    except Exception:
        return rows

    # Try CSV-ish (commas or semicolons)
    if "," in txt or ";" in txt:
        import csv as _csv
        sep = "," if "," in txt else ";"
        for r in _csv.reader(txt.splitlines(), delimiter=sep):
            if not r or len([x for x in r if x.strip() != ""]) < 2:
                continue
            # heuristics for header
            headerish = "".join(r).lower()
            if any(k in headerish for k in ["start", "end", "dur", "label"]) and not any(
                    ch.isdigit() for ch in headerish):
                continue
            vals = [x.strip() for x in r]
            try:
                a = float(vals[0]);
                b = float(vals[1])
                # detect if second is duration
                if a <= b:  # likely start,end
                    start, end = a, b
                else:
                    # if second looks like short vs huge first -> maybe duration
                    start, end = a, a + b
                label = vals[2] if len(vals) >= 3 else ""
                if "seiz" in label.lower():
                    rows.append((start, end, label))
            except Exception:
                continue
        return rows

    # Whitespace-delimited
    for line in txt.splitlines():
        parts = [p for p in re.split(r"\s+", line.strip()) if p]
        if len(parts) < 2:
            continue
        # last token may be label
        try:
            a = float(parts[0]);
            b = float(parts[1])
        except Exception:
            continue
        label = parts[2] if len(parts) >= 3 else ""
        start, end = (a, b) if a <= b else (a, a + b)
        if "seiz" in label.lower():
            rows.append((start, end, label))
    return rows


def tuh_file_rows() -> List[Dict]:
    rows = []
    if not TUH_ROOT.exists():
        return rows
    for f in TUH_ROOT.rglob("*"):
        if not (f.is_file() and is_edf(f)):
            continue
        total = edf_duration_seconds(f)
        sidecar = tuh_find_sidecar(f)
        segs = []
        if sidecar is not None:
            triplets = tuh_parse_sidecar(sidecar)
            for s, e, _lab in triplets:
                segs.append((s, e))
        segs = clip_segments_to_total(segs, total if math.isfinite(total) else float("inf"))
        seiz = sum_durations(segs)
        non = max(0.0, (total - seiz)) if math.isfinite(total) else ""
        rows.append({
            "dataset": "tuh",
            "subject_id": tuh_subject_from_rel(TUH_ROOT, f),
            "file_relpath": str(f.relative_to(TUH_ROOT)),
            "total_seconds": round(total, 6) if math.isfinite(total) else "",
            "seizure_seconds": round(seiz, 6),
            "nonseizure_seconds": round(non, 6) if non != "" else "",
            "n_seizure_events": len(segs),
        })
    return rows


# --------- Aggregation & plotting ---------
def aggregate_per_subject(file_rows: List[Dict]) -> List[Dict]:
    agg = defaultdict(lambda: {"seiz": 0.0, "non": 0.0, "n": 0})
    for r in file_rows:
        k = (r["dataset"], r["subject_id"])
        agg[k]["seiz"] += float(r["seizure_seconds"])
        non = r["nonseizure_seconds"]
        agg[k]["non"] += float(non) if non != "" else 0.0
        agg[k]["n"] += int(r["n_seizure_events"])
    out = []
    for (dataset, subject), v in agg.items():
        out.append({
            "dataset": dataset,
            "subject_id": subject,
            "seizure_hours": round(v["seiz"] / 3600.0, 6),
            "nonseizure_hours": round(v["non"] / 3600.0, 6),
            "n_seizure_events": v["n"],
        })
    return out


def seizure_length_buckets(file_rows: List[Dict]) -> Counter:
    buckets = Counter()
    # Need original segments; we don’t store them—re-parse minimally per file (CHB-MIT & TUH with sidecars)
    # For efficiency we’ll only bucket from per-file rows we can re-derive segments for.
    # CHB-MIT
    for r in file_rows:
        dataset = r["dataset"]
        if dataset == "chbmit":
            f = CHBMIT_ROOT / r["file_relpath"]
            total = float(r["total_seconds"]) if r["total_seconds"] != "" else float("inf")
            segs = clip_segments_to_total(chb_parse_seizures_for_file(f), total)
        elif dataset == "tuh":
            f = TUH_ROOT / r["file_relpath"]
            total = float(r["total_seconds"]) if r["total_seconds"] != "" else float("inf")
            side = tuh_find_sidecar(f)
            segs = []
            if side is not None:
                segs = [(s, e) for s, e, _ in tuh_parse_sidecar(side)]
                segs = clip_segments_to_total(segs, total)
        else:
            continue
        for s, e in segs:
            d = e - s
            if d < 10:
                buckets["<10s"] += 1
            elif d < 30:
                buckets["10-30s"] += 1
            elif d < 60:
                buckets["30-60s"] += 1
            else:
                buckets["≥60s"] += 1
    return buckets


def plot_histogram(values, bins, title, xlabel, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_stacked_bars_per_subject(subj_rows: List[Dict], dataset: str, out_png: Path, top_n: int = 50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # filter dataset
    data = [r for r in subj_rows if r["dataset"] == dataset]
    # sort by total hours desc and keep top_n to keep plot readable
    data.sort(key=lambda x: x["seizure_hours"] + x["nonseizure_hours"], reverse=True)
    data = data[:top_n]
    labels = [r["subject_id"] for r in data]
    non = [r["nonseizure_hours"] for r in data]
    seiz = [r["seizure_hours"] for r in data]

    fig_h = 0.28 * len(data) + 1.8
    fig, ax = plt.subplots(figsize=(10, max(3.5, fig_h)))
    y = list(range(len(data)))
    ax.barh(y, non, label="Non-seizure")
    ax.barh(y, seiz, left=non, label="Seizure")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Hours")
    ax.set_title(f"{dataset.upper()} — Seizure vs Non-seizure hours per subject (top {len(data)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    # Build per-file rows
    rows_chb = chb_file_rows()
    rows_tuh = tuh_file_rows()
    all_rows = rows_chb + rows_tuh

    # Save per-file summary
    per_file_csv = OUTDIR / "label_distribution_per_file.csv"
    with per_file_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "subject_id", "file_relpath", "total_seconds", "seizure_seconds", "nonseizure_seconds",
            "n_seizure_events"
        ])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"Wrote {per_file_csv} with {len(all_rows)} rows")

    # Aggregate per subject
    subj_rows = aggregate_per_subject(all_rows)
    per_subject_csv = OUTDIR / "label_distribution_per_subject.csv"
    with per_subject_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "subject_id", "seizure_hours", "nonseizure_hours", "n_seizure_events"
        ])
        w.writeheader()
        for r in sorted(subj_rows, key=lambda x: (x["dataset"], x["subject_id"])):
            w.writerow(r)
    print(f"Wrote {per_subject_csv} with {len(subj_rows)} subjects")

    # Class imbalance (overall)
    totals = defaultdict(lambda: {"seiz": 0.0, "non": 0.0})
    for r in subj_rows:
        totals[r["dataset"]]["seiz"] += r["seizure_hours"]
        totals[r["dataset"]]["non"] += r["nonseizure_hours"]
    imbalance_csv = OUTDIR / "class_imbalance_summary.csv"
    with imbalance_csv.open("w", newline="") as f:
        w = csv.DictWriter(f,
                           fieldnames=["dataset", "seizure_hours", "nonseizure_hours", "imbalance_ratio_non_to_seiz"])
        w.writeheader()
        for ds, v in totals.items():
            ratio = (v["non"] / v["seiz"]) if v["seiz"] > 0 else float("inf")
            w.writerow({
                "dataset": ds,
                "seizure_hours": round(v["seiz"], 6),
                "nonseizure_hours": round(v["non"], 6),
                "imbalance_ratio_non_to_seiz": round(ratio, 3) if math.isfinite(ratio) else "inf"
            })
    print(f"Wrote {imbalance_csv}")

    # Seizure duration buckets
    buckets = seizure_length_buckets(all_rows)
    buckets_csv = OUTDIR / "seizure_duration_buckets.csv"
    with buckets_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "count"])
        for k in ["<10s", "10-30s", "30-60s", "≥60s"]:
            w.writerow([k, buckets.get(k, 0)])
    print(f"Wrote {buckets_csv}")

    # Plots
    # 1) Histogram of seizure episode lengths (seconds)
    # Rebuild raw list of durations using same logic
    durations = []
    # CHB
    for r in rows_chb:
        f = CHBMIT_ROOT / r["file_relpath"]
        total = float(r["total_seconds"]) if r["total_seconds"] != "" else float("inf")
        for s, e in clip_segments_to_total(chb_parse_seizures_for_file(f), total):
            durations.append(e - s)
    # TUH
    for r in rows_tuh:
        f = TUH_ROOT / r["file_relpath"]
        total = float(r["total_seconds"]) if r["total_seconds"] != "" else float("inf")
        side = tuh_find_sidecar(f)
        if side is not None:
            for s, e, _ in tuh_parse_sidecar(side):
                for (ss, ee) in clip_segments_to_total([(s, e)], total):
                    durations.append(ee - ss)

    if durations:
        plot_histogram(
            values=durations,
            bins=30,
            title="Seizure Duration Distribution (seconds)",
            xlabel="Duration (s)",
            out_png=PLOTDIR / "seizure_duration_hist.png"
        )

    # 2) Stacked bar per subject (CHB & TUH)
    if any(r["dataset"] == "chbmit" for r in subj_rows):
        plot_stacked_bars_per_subject(subj_rows, "chbmit", PLOTDIR / "per_subject_hours_chbmit.png", top_n=50)
    if any(r["dataset"] == "tuh" for r in subj_rows):
        plot_stacked_bars_per_subject(subj_rows, "tuh", PLOTDIR / "per_subject_hours_tuh.png", top_n=50)

    print(f"Plots in {PLOTDIR}")


if __name__ == "__main__":
    main()
