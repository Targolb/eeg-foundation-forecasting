# src/eegff/viz/04_visualization.py
# !/usr/bin/env python
import os, math, csv, re
from pathlib import Path
import pandas as pd
import numpy as np

# Optional dotenv
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

CHBMIT_ROOT = Path(os.getenv("CHBMIT_ROOT", "")).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).expanduser().resolve()
OUT_VIZ = OUTPUT_DIR / "visuals"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

LABEL_SUBJ = OUTPUT_DIR / "labels" / "label_distribution_per_subject.csv"
LABEL_FILE = OUTPUT_DIR / "labels" / "label_distribution_per_file.csv"
SESSION_CSV = OUTPUT_DIR / "sessions" / "session_overview.csv"
QUALITY_CSV = OUTPUT_DIR / "quality" / "signal_quality_summary.csv"


def plot_hours_per_subject():
    df = pd.read_csv(LABEL_SUBJ)
    for ds in ["chbmit", "tuh"]:
        d = df[df.dataset == ds].copy()
        if d.empty:
            continue
        d["total_hours"] = d["seizure_hours"].fillna(0) + d["nonseizure_hours"].fillna(0)
        d = d.sort_values("total_hours", ascending=False)
        # Stacked bars like before but wide figure for readability
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(12, max(3.5, 0.28 * len(d) + 1.8)))
        ax.barh(y, d["nonseizure_hours"], label="Non-seizure")
        ax.barh(y, d["seizure_hours"], left=d["nonseizure_hours"], label="Seizure")
        ax.set_yticks(y)
        ax.set_yticklabels(d["subject_id"], fontsize=8)
        ax.set_xlabel("Hours")
        ax.set_title(f"{ds.upper()} — Total hours per subject")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_VIZ / f"hours_per_subject_bar_{ds}.png", dpi=170)
        plt.close(fig)


def plot_seizure_counts_per_subject_chbmit():
    df = pd.read_csv(LABEL_SUBJ)
    d = df[df.dataset == "chbmit"].copy()
    if d.empty:
        return
    d = d.sort_values("n_seizure_events", ascending=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(d["subject_id"], d["n_seizure_events"])
    ax.set_ylabel("# seizures")
    ax.set_xlabel("Subject")
    ax.set_title("CHB-MIT — Seizure counts per subject")
    ax.tick_params(axis='x', rotation=60)
    fig.tight_layout()
    fig.savefig(OUT_VIZ / "seizure_counts_per_subject_chbmit.png", dpi=170)
    plt.close(fig)


def plot_channel_variance_heatmap():
    """Heatmap of channel variance across dataset using QUALITY_CSV."""
    if not QUALITY_CSV.exists():
        return
    q = pd.read_csv(QUALITY_CSV)
    # Keep EEG channels only (drop obvious aux)
    bad_patterns = ["EKG", "EMG", "DC", "PHOTIC", "IBI", "BURSTS", "SUPPR", "PG", "ROC", "LOC"]
    keep = ~q["channel"].astype(str).str.contains("|".join(bad_patterns), case=False, regex=True)
    q = q[keep].copy()
    # Pivot: rows=channel, cols=dataset:file (small cap), values=variance
    q["file_key"] = q["dataset"].str.upper() + ":" + q["file_relpath"].astype(str)
    pivot = q.pivot_table(index="channel", columns="file_key", values="var_uv2", aggfunc="median")
    # log10 for dynamic range
    mat = np.log10(pivot.replace(0, np.nan))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=6)
    ax.set_xticks([])
    ax.set_title("Channel variance heatmap (log10 var)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(var $\\mu V^2$)")
    fig.tight_layout()
    fig.savefig(OUT_VIZ / "channel_variance_heatmap.png", dpi=170)
    plt.close(fig)


# ---- Helpers to pull example traces ----
def _chb_parse_seizures_for_file(edf_path: Path):
    segs = []
    parent = edf_path.parent
    stem = edf_path.stem
    base = stem[:-4] if stem.lower().endswith(".edf") else stem
    # <file>.seizures
    fz = parent / f"{base}.seizures"
    if fz.exists():
        txt = fz.read_text(errors="ignore")
        for m in re.finditer(r"start\s*time\s*:\s*([0-9.]+)\s*.*?end\s*time\s*:\s*([0-9.]+)", txt, flags=re.I | re.S):
            segs.append((float(m.group(1)), float(m.group(2))))
    # summary text near the file
    for summ in list(parent.glob("*summary*.txt")) + list(parent.parent.glob("*summary*.txt")):
        try:
            txt = summ.read_text(errors="ignore")
        except Exception:
            continue
        base_re = re.compile(re.escape(base), re.I)
        for hit in base_re.finditer(txt):
            win = txt[hit.start():hit.start() + 2000]
            for m in re.finditer(r"start\s*time\s*:\s*([0-9.]+)\s*.*?end\s*time\s*:\s*([0-9.]+)", win,
                                 flags=re.I | re.S):
                segs.append((float(m.group(1)), float(m.group(2))))
    # merge overlaps
    if not segs:
        return []
    segs.sort()
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def plot_example_traces():
    """One CHB-MIT example: 10s seizure vs 10s non-seizure from same file."""
    if not (LABEL_FILE.exists() and CHBMIT_ROOT.exists()):
        return
    df = pd.read_csv(LABEL_FILE)
    cand = df[(df.dataset == "chbmit") & (df.n_seizure_events > 0)]
    if cand.empty:
        return
    row = cand.iloc[0]
    f = CHBMIT_ROOT / row.file_relpath
    # parse seizure intervals
    segs = _chb_parse_seizures_for_file(f)
    if not segs:
        return
    s0, e0 = segs[0]
    # choose 10s windows
    sz_start = max(0.0, s0 + 1.0)  # a bit inside
    sz_stop = min(e0, sz_start + 10.0)
    non_start = max(0.0, s0 - 20.0)  # 10–20s before seizure
    non_stop = non_start + 10.0
    import mne
    raw = mne.io.read_raw_edf(str(f), preload=True, verbose="ERROR")
    # Pick a few standard channels if present
    picks = [ch for ch in ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FZ-CZ", "CZ-PZ"]
             if ch in raw.ch_names]
    if not picks:
        picks = raw.ch_names[:8]
    raw.pick(picks)
    # crop & plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1 = raw.copy().crop(sz_start, sz_stop).plot(n_channels=len(picks), scalings="auto", show=False)
    fig1.suptitle(f"Seizure snippet ({Path(row.file_relpath).name}, {sz_stop - sz_start:.1f}s)")
    fig1.savefig(OUT_VIZ / "example_traces_seizure.png", dpi=160)

    fig2 = raw.copy().crop(non_start, non_stop).plot(n_channels=len(picks), scalings="auto", show=False)
    fig2.suptitle(f"Non-seizure snippet ({Path(row.file_relpath).name}, {non_stop - non_start:.1f}s)")
    fig2.savefig(OUT_VIZ / "example_traces_nonseizure.png", dpi=160)

    # side-by-side quick comparison
    fig, ax = plt.subplots(figsize=(9, 0.01))  # dummy so the two figs exist; we already saved above
    plt.close(fig)


def main():
    print("Making visuals…")
    plot_hours_per_subject()
    plot_seizure_counts_per_subject_chbmit()
    plot_channel_variance_heatmap()
    plot_example_traces()
    print(f"Done. See {OUT_VIZ}")


if __name__ == "__main__":
    main()
