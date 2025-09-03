# src/utils/canonical_channels.py
CANONICAL_1020 = {
    "FP1","FP2","F3","F4","F7","F8","FZ","T3","T4","T5","T6","C3","C4","CZ",
    "P3","P4","PZ","O1","O2","A1","A2","T1","T2","FC1","FC2","CP1","CP2"
}

def norm_ch(name: str) -> str:
    if not name: return ""
    x = name.upper().strip()
    for suf in ["-REF","-REF.","_REF","-LE","-RE","(REF)"]:
        x = x.replace(suf, "")
    x = x.replace("EEG ", "").replace(" ", "")
    return x
