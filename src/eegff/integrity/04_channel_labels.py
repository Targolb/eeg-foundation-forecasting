#!/usr/bin/env python
import csv, argparse, random, collections
from pathlib import Path
from src.eegff.utils.path_utils import CHBMIT_ROOT, TUH_ROOT, KAGGLE_EEG_ROOT, OUTPUT_DIR
from src.eegff.utils.canonical_channels import CANONICAL_1020, norm_ch

def edf_channels(path: Path):
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    return [ch["ch_name"] for ch in raw.info["chs"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=80)
    ap.add_argument("--outdir", default=str(OUTPUT_DIR / "integrity"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    def sample(root, exts, n):
        files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
        return files if n<=0 or n>=len(files) else random.sample(files, n)

    counters = collections.Counter()
    unknowns = collections.Counter()

    for ds, root in [("chbmit", CHBMIT_ROOT), ("tuh(edf)", TUH_ROOT)]:
        for p in sample(root, {".edf"}, args.sample_n):
            for ch in edf_channels(p):
                n = norm_ch(ch)
                counters[(ds,n)] += 1
                if n and n not in CANONICAL_1020:
                    unknowns[(ds,n)] += 1

    # write seen channels
    with open(outdir / "channels_seen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","channel","count"])
        w.writeheader()
        for (ds,ch), c in sorted(counters.items()):
            w.writerow({"dataset":ds,"channel":ch,"count":c})

    # write unknowns + naive suggestions
    def suggest(n):
        # simple suggestions for common variants
        n2 = n.replace("FPZ","FPZ").replace("CZ","CZ")
        return n2 if n2 in CANONICAL_1020 else ""

    with open(outdir / "channels_unknown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","raw_name","suggestion"])
        w.writeheader()
        for (ds,ch), _ in sorted(unknowns.items()):
            w.writerow({"dataset":ds,"raw_name":ch,"suggestion":suggest(ch)})

    print("Channel label reports written.")

if __name__ == "__main__":
    main()
