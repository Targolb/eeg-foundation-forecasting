#!/usr/bin/env python
import csv, argparse, re
from pathlib import Path
from eegff.utils.path_utils import CHBMIT_ROOT, TUH_ROOT, KAGGLE_EEG_ROOT, OUTPUT_DIR

EDF_PATTERNS = (".edf", ".edf.gz")


# ---- helpers ----
def list_files(root: Path, patterns):
    files = []
    if not root or not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and any(p.name.lower().endswith(ptn) for ptn in patterns):
            files.append(p)
    return files


# parse TUH filename like: aaaaaanr_s001_t001.edf
TUH_NAME_RE = re.compile(r"^(?P<subj>[a-z0-9]+)_s(?P<sess>\d+)_t(?P<trial>\d+)\.edf(\.gz)?$", re.I)


def parse_tuh_row(root: Path, file_path: Path):
    rel = file_path.relative_to(root)
    parts = rel.parts
    # expected: ["00_epilepsy"|"01_no_epilepsy", "<subject>", "<montage>", "<file>"]
    cohort = parts[0] if len(parts) >= 1 else ""
    subject = parts[1] if len(parts) >= 2 else ""
    montage = parts[2] if len(parts) >= 3 else ""
    fname = parts[-1]
    m = TUH_NAME_RE.match(fname.lower())
    session_id = m.group("sess") if m else ""
    trial_id = m.group("trial") if m else ""
    return {
        "relpath": str(rel),
        "cohort": cohort,  # 00_epilepsy / 01_no_epilepsy
        "subject_id": subject,  # aaaaaanr
        "montage": montage,  # 02_tcp_le
        "session_id": session_id,  # 001
        "trial_id": trial_id,  # 001
        "size_bytes": file_path.stat().st_size,
    }


def parse_chb_row(root: Path, file_path: Path):
    rel = file_path.relative_to(root)
    return {"relpath": str(rel), "size_bytes": file_path.stat().st_size}


def parse_kaggle_row(root: Path, file_path: Path):
    rel = file_path.relative_to(root)
    return {"relpath": str(rel), "size_bytes": file_path.stat().st_size}


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(OUTPUT_DIR / "integrity"))
    args = ap.parse_args()
    outdir = Path(args.outdir);
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- CHB-MIT ----
    chb_files = list_files(CHBMIT_ROOT, EDF_PATTERNS)
    chb_rows = [parse_chb_row(CHBMIT_ROOT, p) for p in chb_files]
    write_csv(outdir / "manifest_chbmit.csv",
              chb_rows,
              ["relpath", "size_bytes"])

    # ---- TUH ----
    tuh_files = list_files(TUH_ROOT, EDF_PATTERNS + (".tfrecord",))
    tuh_rows = []
    for p in tuh_files:
        name = p.name.lower()
        if name.endswith(".edf") or name.endswith(".edf.gz"):
            tuh_rows.append(parse_tuh_row(TUH_ROOT, p))
        else:
            # minimal info for non-EDF, keep track anyway
            rel = str(p.relative_to(TUH_ROOT))
            tuh_rows.append(
                {"relpath": rel, "cohort": "", "subject_id": "", "montage": "", "session_id": "", "trial_id": "",
                 "size_bytes": p.stat().st_size})
    write_csv(outdir / "manifest_tuh.csv",
              tuh_rows,
              ["relpath", "cohort", "subject_id", "montage", "session_id", "trial_id", "size_bytes"])

    # ---- Kaggle ----
    kag_files = list_files(KAGGLE_EEG_ROOT, (".csv",))
    kag_rows = [parse_kaggle_row(KAGGLE_EEG_ROOT, p) for p in kag_files]
    write_csv(outdir / "manifest_kaggle.csv",
              kag_rows,
              ["relpath", "size_bytes"])

    print("File counts:", {"chbmit": len(chb_rows), "tuh": len(tuh_rows), "kaggle": len(kag_rows)})


if __name__ == "__main__":
    main()
