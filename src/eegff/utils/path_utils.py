# src/utils/path_utils.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHBMIT_ROOT = Path(os.getenv("CHBMIT_ROOT", ""))
TUH_ROOT = Path(os.getenv("TUH_ROOT", ""))
KAGGLE_EEG_ROOT = Path(os.getenv("KAGGLE_EEG_ROOT", ""))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
