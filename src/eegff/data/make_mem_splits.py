import json, re, random
from pathlib import Path
import os
SEED = 1337
random.seed(SEED)

# ---- EDIT THESE ROOTS IF NEEDED ----
CHBMIT_ROOT = Path("/scratch0/Targol/chbmit")
TUH_ROOT = Path("/scratch0/Targol/tuh_eeg_epilepsy_v2.0.1")
KAGGLE_ROOT = Path("/scratch0/Targol/kaggle_eeg")

# IO spec (keep in sync with configs/io_spec.yaml)
IO_SPEC = {
    "channels_common": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                        "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"],
    "channel_aliases": {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"},
    "fs_target": 256,
    "window_s": 5.0,
    "stride_s": 2.5,
}


def list_chbmit_subjects(root: Path):
    return sorted([p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"chb\d+", p.name)])


def list_tuh_subjects(root: Path):
    subs = set()
    for p in root.rglob("*"):
        if p.is_file():
            m = re.search(r"(?:^|[_/])s(\d+)", str(p))
            if m:
                subs.add("s" + m.group(1))
    return sorted(subs)


def list_kaggle_subjects(root: Path):
    subs = set()
    for p in root.iterdir():
        if p.is_dir() and re.fullmatch(r"s\d{2,}", p.name):
            subs.add(p.name)
        elif p.is_file():
            m = re.match(r"(s\d{2,})", p.stem)
            if m:
                subs.add(m.group(1))
    for p in root.rglob("*.csv"):
        m = re.match(r"(s\d{2,})", p.stem)
        if m:
            subs.add(m.group(1))
    return sorted(subs)


def split_subjects(subjects, n_val=2):
    subs = subjects[:]
    random.shuffle(subs)
    n_val = min(n_val, max(1, len(subs) // 10)) if n_val is None else n_val
    val = sorted(subs[:n_val])
    train = sorted(subs[n_val:])
    return train, val


def main():
    # allow env overrides from docker -e VAR=...
    chb_root = Path(os.getenv("CHBMIT_ROOT", "/scratch0/Targol/chbmit"))
    tuh_root = Path(os.getenv("TUH_ROOT", "/scratch0/Targol/tuh_eeg_epilepsy_v2.0.1"))
    kag_root = Path(os.getenv("KAGGLE_ROOT", "/scratch0/Targol/kaggle_eeg"))

    chb_subs = list_chbmit_subjects(chb_root)
    tuh_subs = list_tuh_subjects(tuh_root)
    kag_subs = list_kaggle_subjects(kag_root)

    train_chb, val_chb = split_subjects(chb_subs, n_val=2)
    train_tuh, val_tuh = split_subjects(tuh_subs, n_val=2)
    train_kag, val_kag = split_subjects(kag_subs, n_val=2)

    out = {
        "spec": {
            "fs": IO_SPEC["fs_target"],
            "window_s": IO_SPEC["window_s"],
            "stride_s": IO_SPEC["stride_s"],
            "channels_common": IO_SPEC["channels_common"],
            "channel_aliases": IO_SPEC["channel_aliases"],
            "seed": SEED,
        },
        "datasets": {
            "chbmit": {"root": str(chb_root), "train_subjects": train_chb, "val_subjects": val_chb},
            "tuh": {"root": str(tuh_root), "train_subjects": train_tuh, "val_subjects": val_tuh},
            "kaggle": {"root": str(kag_root), "train_subjects": train_kag, "val_subjects": val_kag},
        },
    }

    # === Save to project root: /workspace/splits/mem_trainval.json ===
    # file is at /workspace/src/eegff/data/make_mem_splits.py
    # parents[0]=.../data, [1]=.../eegff, [2]=.../src, [3]=.../workspace
    project_root = Path(__file__).resolve().parents[3]
    splits_dir = project_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_path = splits_dir / "mem_trainval.json"

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Wrote {out_path}")
    print(f"Counts -> CHBMIT: {len(chb_subs)}  TUH: {len(tuh_subs)}  Kaggle: {len(kag_subs)}")
    print(f"Val subjects -> CHBMIT: {val_chb}  TUH: {val_tuh}  Kaggle: {val_kag}")


if __name__ == "__main__":
    main()
