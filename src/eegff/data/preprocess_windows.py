import os, json, re, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import iirnotch, filtfilt, butter, resample_poly
import mne  # EDF reading (CHB-MIT)

# ============================= Config =============================

@dataclass
class IOSpec:
    channels_common: List[str]
    channel_aliases: Dict[str, str]
    fs_target: int
    window_s: float
    stride_s: float

@dataclass
class FilterSpec:
    hp_hz: Optional[float] = 0.5
    lp_hz: Optional[float] = 40.0
    notch_hz: Optional[float] = 60.0
    q_notch: float = 30.0

# CHB-MIT canonical bipolar montage (18 pairs)
CHBMIT_BIPOLAR18 = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
]

# ============================== Utils =============================

def _zscore(win: np.ndarray, eps=1e-8):
    mu = win.mean(axis=1, keepdims=True)
    sd = win.std(axis=1, keepdims=True)
    return (win - mu) / (sd + eps)

def _butter_bandpass(hp, lp, fs, order=4):
    if hp is None and lp is None:
        return None, None
    nyq = 0.5 * fs
    if hp is None:
        Wn = lp / nyq
        b, a = butter(order, Wn, btype="lowpass")
    elif lp is None:
        Wn = hp / nyq
        b, a = butter(order, Wn, btype="highpass")
    else:
        Wn = [hp / nyq, lp / nyq]
        b, a = butter(order, Wn, btype="bandpass")
    return b, a

def _apply_filters(x: np.ndarray, fs: int, fcfg: FilterSpec):
    x = x - x.mean(axis=1, keepdims=True)  # DC
    if fcfg.notch_hz is not None and fcfg.notch_hz > 0:
        b_notch, a_notch = iirnotch(fcfg.notch_hz / (fs / 2), fcfg.q_notch)
        x = filtfilt(b_notch, a_notch, x, axis=1, method="gust")
    b, a = _butter_bandpass(fcfg.hp_hz, fcfg.lp_hz, fs)
    if b is not None:
        x = filtfilt(b, a, x, axis=1, method="gust")
    return x

def _resample(x: np.ndarray, fs_in: int, fs_out: int):
    if fs_in == fs_out:
        return x
    from math import gcd
    g = gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    return resample_poly(x, up=up, down=down, axis=1)

def _iter_windows(n_samples: int, fs: int, win_s: float, stride_s: float) -> Iterator[Tuple[int,int]]:
    w = int(round(win_s * fs))
    s = int(round(stride_s * fs))
    start = 0
    while start + w <= n_samples:
        yield start, start + w
        start += s

def _normalize_label(nm: str) -> str:
    n = nm.strip().upper().replace(" ", "")
    n = n.replace("EEG", "")
    n = n.replace("-LE", "").replace("-REF", "")
    n = n.strip("-")
    return n

def _map_channels_try(raw_ch: List[str], target_list: List[str], aliases: Dict[str, str]):
    ali = {k.lower(): v for k, v in aliases.items()} if aliases else {}
    norm = []
    for ch in raw_ch:
        nm = _normalize_label(ch)
        if nm.lower() in ali:
            nm = _normalize_label(ali[nm.lower()])
        norm.append(nm)
    pos = {}
    for i, nm in enumerate(norm):
        if nm not in pos:
            pos[nm] = i
    sel, missing = [], []
    for tgt in target_list:
        tn = _normalize_label(tgt)
        if tn in pos: sel.append(pos[tn])
        else: missing.append(tgt)
    return sel, missing

# ============================== IO ===============================

def _list_chbmit_files(root: Path, subject_id: str):
    subj_dir = root / subject_id
    return sorted(subj_dir.glob("*.edf")) if subj_dir.exists() else []

def _load_chbmit_edf(path: Path):
    try:
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    except Exception as e:
        raise RuntimeError(f"EDF read failed for {path}: {e}")
    fs = int(round(raw.info["sfreq"]))
    data = raw.get_data() * 1e6  # V → µV
    return data, fs, raw.info["ch_names"], path.stem

# =========================== Writer / QA ==========================

class ShardWriter:
    """
    Keyed writer: keeps separate buffers per (dataset, subject_id, channels_used)
    so (C,T) stays consistent within each shard.
    """
    def __init__(self, out_root: Path, shard_size: int):
        self.out_root = out_root
        self.shard_size = shard_size
        # key -> dict(x: list, meta: list, channels: list)
        self._bufs: Dict[Tuple[str,str,str], Dict[str, list]] = {}

    def _flush_key(self, key):
        buf = self._bufs.get(key)
        if not buf or not buf["x"]:
            return
        ds = buf["meta"][0]["dataset"]
        sbj = buf["meta"][0]["subject_id"]
        chset = buf["meta"][0]["channels_used"]  # 'monopolar19' or 'bipolar18'
        out_dir = self.out_root / ds / sbj / chset
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(out_dir.glob("shard_*.npz")))
        path = out_dir / f"shard_{idx:05d}.npz"
        np.savez_compressed(
            path,
            x=np.stack(buf["x"], axis=0),
            meta=np.array(buf["meta"], dtype=object),
            channels=np.array(buf["channels"], dtype=object),
        )
        buf["x"].clear(); buf["meta"].clear()

    def add(self, x: np.ndarray, meta: Dict, channels: List[str]):
        key = (meta["dataset"], meta["subject_id"], meta["channels_used"])
        if key not in self._bufs:
            self._bufs[key] = {"x": [], "meta": [], "channels": channels}
        self._bufs[key]["x"].append(x)
        self._bufs[key]["meta"].append(meta)
        if len(self._bufs[key]["x"]) >= self.shard_size:
            self._flush_key(key)

    def close(self):
        for key in list(self._bufs.keys()):
            self._flush_key(key)

class QALogger:
    def __init__(self): self.rows = []
    def log(self, **kw): self.rows.append(kw)
    def save(self, path: Path):
        df = pd.DataFrame(self.rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            md = "# Pretrain Data QC\n\n" + df.head(200).to_markdown(index=False)
        except Exception:
            md = "# Pretrain Data QC (CSV fallback)\n\n" + df.head(200).to_csv(index=False)
        path.write_text(md)

# ========================== Processing ===========================

def process_subject_chbmit(root: Path, subject_id: str, io: IOSpec, fcfg: FilterSpec,
                           writer: ShardWriter, qa: QALogger):
    files = _list_chbmit_files(root, subject_id)
    if not files:
        print(f"[WARN] No EDFs for {subject_id} at {root/subject_id}")
        qa.log(dataset="chbmit", subject=subject_id, session="", issue="no_files", missing="")
        return

    for f in files:
        try:
            x_uv, fs_in, ch_raw, session_id = _load_chbmit_edf(f)
        except Exception as e:
            print(f"[ERR] {e}")
            qa.log(dataset="chbmit", subject=subject_id, session=f.name, issue=f"read_error:{e}", missing="")
            continue

        # Try monopolar (19) first
        sel, missing = _map_channels_try(ch_raw, io.channels_common, io.channel_aliases)
        used_set = "monopolar19"
        target_channels = io.channels_common

        # If no match, try CHB-MIT bipolar (18)
        if len(sel) == 0:
            sel_bp, missing_bp = _map_channels_try(ch_raw, CHBMIT_BIPOLAR18, {})
            if len(sel_bp) == 0:
                qa.log(dataset="chbmit", subject=subject_id, session=f.name,
                       issue="no_common_channels", missing=",".join(missing_bp))
                print(f"[WARN] no_common_channels for {subject_id}/{f.name}")
                continue
            sel = sel_bp
            used_set = "bipolar18"
            target_channels = CHBMIT_BIPOLAR18
            missing = missing_bp

        x_ct = x_uv[sel, :]
        x_ct = _resample(x_ct, fs_in, io.fs_target)
        fs = io.fs_target
        x_ct = _apply_filters(x_ct, fs, fcfg)

        W = int(round(io.window_s * fs))
        for s0, s1 in _iter_windows(x_ct.shape[1], fs, io.window_s, io.stride_s):
            win = x_ct[:, s0:s1]
            if win.shape[1] != W: continue
            if float(np.median(win.std(axis=1))) < 1e-3: continue
            win_z = _zscore(win)
            meta = {
                "dataset": "chbmit",
                "subject_id": subject_id,
                "session_id": session_id,
                "start_t": float(s0 / fs),
                "fs": fs,
                "channels": target_channels,
                "channels_used": used_set,
            }
            writer.add(win_z.astype(np.float32), meta, target_channels)

        qa.log(dataset="chbmit", subject=subject_id, session=f.name, issue="ok",
               missing=",".join(missing))

    print(f"[OK] {subject_id}: processed {len(files)} EDF(s)")

# ============================= Main ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="splits/mem_trainval.json")
    ap.add_argument("--out", default="outputs/preprocessing", help="output root for shards")
    ap.add_argument("--qa_md", default="outputs/reports/pretrain_data_qc.md")
    ap.add_argument("--datasets", default="chbmit", help="comma list: chbmit,tuh,kaggle")
    ap.add_argument("--include_val", action="store_true")
    ap.add_argument("--bandpass", default="0.5,40", help="hp,lp or 'none,none'")
    ap.add_argument("--notch", type=float, default=60.0, help="<=0 to disable")
    ap.add_argument("--shard_size", type=int, default=2048)
    ap.add_argument("--subjects", type=str, default="", help="comma-separated subject IDs to override splits")
    args = ap.parse_args()

    cfg = json.loads(Path(args.splits).read_text())
    io = IOSpec(
        channels_common=cfg["spec"]["channels_common"],
        channel_aliases=cfg["spec"]["channel_aliases"],
        fs_target=int(cfg["spec"]["fs"]),
        window_s=float(cfg["spec"]["window_s"]),
        stride_s=float(cfg["spec"]["stride_s"]),
    )

    # Filters
    if args.bandpass.lower() in ("none", "none,none"):
        hp, lp = None, None
    else:
        hp_s, lp_s = args.bandpass.split(",")
        hp = None if hp_s.strip().lower()=="none" else float(hp_s)
        lp = None if lp_s.strip().lower()=="none" else float(lp_s)
    notch = None if args.notch is None or args.notch <= 0 else float(args.notch)
    fcfg = FilterSpec(hp_hz=hp, lp_hz=lp, notch_hz=notch)

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    qa_path = Path(args.qa_md); qa_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_root, args.shard_size)
    qa = QALogger()

    want = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}
    subjects_override = [s.strip() for s in args.subjects.split(",") if s.strip()]

    # CHB-MIT
    if "chbmit" in want:
        chb_root = Path(cfg["datasets"]["chbmit"]["root"])
        subs = subjects_override or cfg["datasets"]["chbmit"]["train_subjects"][:]
        if args.include_val and not subjects_override:
            subs += cfg["datasets"]["chbmit"]["val_subjects"]
        for sid in tqdm(sorted(subs), desc="CHB-MIT"):
            process_subject_chbmit(chb_root, sid, io, fcfg, writer, qa)

    # TODO: add TUH and Kaggle loaders

    writer.close()
    qa.save(qa_path)
    print(f"[DONE] Shards -> {out_root}")
    print(f"[DONE] QA report -> {qa_path}")

if __name__ == "__main__":
    main()
