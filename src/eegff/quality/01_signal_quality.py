# src/eegff/quality/01_signal_quality.py
# !/usr/bin/env python
import os
import math
import csv
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Optional dotenv (safe if not installed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ---------- Config via env (or -e in Docker) ----------
CHBMIT_ROOT = Path(os.getenv("CHBMIT_ROOT", "")).expanduser().resolve()
TUH_ROOT = Path(os.getenv("TUH_ROOT", "")).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).expanduser().resolve()
OUTDIR = OUTPUT_DIR / "quality"
PLOTDIR = OUTDIR / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def is_edf(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".edf") or n.endswith(".edf.gz")


def list_edfs(root: Path) -> List[Path]:
    if not root or not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and is_edf(p)]


def take_sample(paths: List[Path], k: int, seed: int = 7) -> List[Path]:
    if k <= 0 or k >= len(paths):
        return paths
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(paths), size=k, replace=False)
    return [paths[i] for i in idx]


def edf_meta_and_data(path: Path, max_seconds: float, picks=None) -> Tuple[float, float, np.ndarray, List[str]]:
    """
    Returns: (sfreq, duration_sec, data_uv, ch_names)
    - Loads up to max_seconds (from beginning) without preloading full file.
    - Converts to microvolts (µV).
    """
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, exclude=[])
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    duration_sec = n_times / sfreq if sfreq > 0 else float("nan")
    # slice
    n_take = int(min(n_times, max_seconds * sfreq))
    data = raw.get_data(picks=picks, start=0, stop=n_take)  # shape (n_ch, n_take)
    ch_names = [raw.ch_names[i] for i in picks]
    # convert to µV (MNE returns in Volts for EEG)
    data_uv = data * 1e6
    return sfreq, duration_sec, data_uv, ch_names


def channel_stats(x_uv: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Per-channel stats for a 1D array in µV.
    Returns: (mean_uv, var_uv2, frac_missing, ptp_uv)
    """
    if x_uv.size == 0:
        return (np.nan, np.nan, 1.0, 0.0)
    isnan = np.isnan(x_uv)
    frac_missing = float(isnan.mean())
    if frac_missing >= 1.0:
        return (np.nan, np.nan, 1.0, 0.0)
    x = x_uv[~isnan]
    mean = float(np.mean(x))
    var = float(np.var(x))
    ptp = float(np.ptp(x))  # peak-to-peak
    return (mean, var, frac_missing, ptp)


def is_flatline(std_uv: float, ptp_uv: float, var_uv2: float) -> bool:
    """
    Heuristics for flatline/dead channel.
    - Very small std and ptp and variance.
    """
    if not np.isfinite(std_uv) or not np.isfinite(ptp_uv) or not np.isfinite(var_uv2):
        return False
    return (std_uv < 0.5) and (ptp_uv < 1.0) and (var_uv2 < 1e-1)  # thresholds in µV/µV^2


def extreme_flag(x_uv: np.ndarray, threshold_uv: float, max_frac: float = 0.01) -> Tuple[bool, float]:
    """
    Mark channel as extreme if fraction of samples exceeding |threshold_uv|
    is greater than max_frac (default 1%).
    Returns: (flag, frac_extreme)
    """
    if x_uv.size == 0:
        return False, 0.0
    isnan = np.isnan(x_uv)
    if isnan.all():
        return False, 0.0
    x = np.abs(x_uv[~isnan])
    frac = float((x > threshold_uv).mean()) if x.size else 0.0
    return (frac > max_frac), frac


def plot_snippet(path: Path, data_uv: np.ndarray, sfreq: float, ch_names: List[str], seconds: float, out_png: Path):
    """
    Plot first 'seconds' of data for up to 12 channels stacked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ch, n_t = data_uv.shape
    n_plot = min(n_ch, 12)
    n_take = int(min(n_t, seconds * sfreq))
    t = np.arange(n_take) / sfreq

    fig_h = max(2.5, 0.35 * n_plot + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    offset = 0.0
    step = 120.0  # µV offset between channels
    for i in range(n_plot):
        y = data_uv[i, :n_take] + offset
        ax.plot(t, y, linewidth=0.6)
        ax.text(t[0] if len(t) else 0, offset, ch_names[i], va="bottom", fontsize=8)
        offset += step

    ax.set_title(f"EEG snippet: {path.name} (first {min(seconds, n_take / sfreq):.1f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV, stacked)")
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------- Main routine ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=5, help="Files to sample per dataset (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=600.0, help="Max seconds to analyze per file")
    ap.add_argument("--plot", action="store_true", help="Save small raw plots for sampled files")
    ap.add_argument("--plot-seconds", type=float, default=20.0, help="Seconds to show in plots")
    ap.add_argument("--extreme-thresh-uv", type=float, default=500.0, help="Extreme amplitude threshold (µV)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Collect files
    chb_files = list_edfs(CHBMIT_ROOT)
    tuh_files = list_edfs(TUH_ROOT)

    chb_take = take_sample(chb_files, args.n_samples, args.seed)
    tuh_take = take_sample(tuh_files, args.n_samples, args.seed)

    out_csv = OUTDIR / "signal_quality_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "subject_id", "file_relpath", "sfreq", "duration_hours",
            "channel", "mean_uv", "var_uv2", "std_uv", "ptp_uv", "frac_missing",
            "is_flatline", "is_extreme", "frac_extreme"
        ])
        w.writeheader()

        # Process a list of EDFs
        def process_many(dataset: str, root: Path, files: List[Path]):
            for p in files:
                try:
                    sfreq, dur_sec, data_uv, ch_names = edf_meta_and_data(p, args.max_seconds)
                except Exception as e:
                    # Skip unreadable files
                    continue

                rel = str(p.relative_to(root)) if p.is_relative_to(root) else p.name
                subject_id = ""
                # subject from path: chbXX for CHB; TUH subject is second folder element
                if dataset == "chbmit":
                    for part in p.parts:
                        if part.lower().startswith("chb"):
                            subject_id = part
                            break
                elif dataset == "tuh":
                    relp = p.relative_to(root)
                    if len(relp.parts) >= 2:
                        subject_id = relp.parts[1]

                duration_hours = dur_sec / 3600.0 if np.isfinite(dur_sec) else ""

                for i, ch in enumerate(ch_names):
                    x = data_uv[i]
                    mean_uv, var_uv2, frac_missing, ptp_uv = channel_stats(x)
                    std_uv = float(np.sqrt(var_uv2)) if np.isfinite(var_uv2) and var_uv2 >= 0 else np.nan
                    flat = is_flatline(std_uv, ptp_uv, var_uv2)
                    ext_flag, frac_extreme = extreme_flag(x, args.extreme_thresh_uv)

                    w.writerow({
                        "dataset": dataset,
                        "subject_id": subject_id,
                        "file_relpath": rel,
                        "sfreq": round(sfreq, 3),
                        "duration_hours": round(duration_hours, 6) if duration_hours != "" else "",
                        "channel": ch,
                        "mean_uv": round(mean_uv, 6) if np.isfinite(mean_uv) else "",
                        "var_uv2": round(var_uv2, 6) if np.isfinite(var_uv2) else "",
                        "std_uv": round(std_uv, 6) if np.isfinite(std_uv) else "",
                        "ptp_uv": round(ptp_uv, 6) if np.isfinite(ptp_uv) else "",
                        "frac_missing": round(frac_missing, 6) if np.isfinite(frac_missing) else "",
                        "is_flatline": int(flat),
                        "is_extreme": int(ext_flag),
                        "frac_extreme": round(frac_extreme, 6),
                    })

                # Optional plots (first 12 channels)
                if args.plot:
                    png = PLOTDIR / f"{dataset}__{Path(rel).name}.png"
                    try:
                        plot_snippet(p, data_uv, sfreq, ch_names, args.plot_seconds, png)
                    except Exception:
                        pass

        process_many("chbmit", CHBMIT_ROOT, chb_take)
        process_many("tuh", TUH_ROOT, tuh_take)

    print(f"Wrote {out_csv}")
    if args.plot:
        print(f"Plots in {PLOTDIR}")


if __name__ == "__main__":
    main()
