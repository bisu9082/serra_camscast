"""Number-to-source audit.

Every numeric literal in the manuscript and the supplement is extracted with the
sentence it sits in, and matched against the computed artefacts. The point is not
to prove every number right automatically: it is to produce a short list of
numbers that no artefact contains, which is where a transcription error such as
an ETS quoted from the wrong column will surface.

Writes number_audit.csv (full) and number_audit_unmatched.csv (the review list).
"""
import json, re, glob, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- artefacts
values = {}          # rounded value -> set of source labels


def add(v, label):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return
    if not np.isfinite(f):
        return
    for nd in (0, 1, 2, 3, 4):
        values.setdefault(round(f, nd), set()).add(label)


def walk(o, label):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{label}.{k}")
    elif isinstance(o, (list, tuple)):
        for v in o:
            walk(v, label)
    elif isinstance(o, (int, float)):
        add(o, label)
        # percentages and their log-ratio partner are the same fact on two scales
        if isinstance(o, float) and -1 < o < 1 and o != 0:
            add(100 * (1 - np.exp(-o)), label + "~pct")
        if isinstance(o, (int, float)) and -100 < o < 100 and o != 0:
            try:
                add(-np.log(1 - o / 100), label + "~delta")
            except (ValueError, ZeroDivisionError):
                pass


for f in sorted(glob.glob(os.path.join(RUN, "*.json"))):
    try:
        walk(json.load(open(f)), os.path.basename(f))
    except Exception as e:
        print("skip", f, e)

for f in sorted(glob.glob(os.path.join(ROOT, "data_processed", "aligned_*.csv"))) + sorted(glob.glob(os.path.join(RUN, "per_point", "*.csv"))):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    lab = os.path.basename(f)
    for c in d.columns:
        s = pd.to_numeric(d[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        for v in (s.min(), s.max(), s.mean(), s.median(), s.sum(), len(s)):
            add(v, f"{lab}:{c}")
    add(len(d), f"{lab}:nrow")

print(f"artefact value index: {len(values)} distinct rounded values", flush=True)

# ---------------------------------------------------------------- manuscript
NUM = re.compile(r"(?<![\w.\-])-?\d+(?:\{,\})?\d*(?:\.\d+)?(?![\w.])")
STRIP = re.compile(r"\\(?:label|ref|cite[tp]?|citep|citet|documentclass|usepackage|input|include)\{[^}]*\}")

# numbers that are structural rather than empirical
IGNORE = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
          "15", "16", "19", "24", "25", "50", "95", "100", "1000", "2000",
          "2016", "2022", "2023", "2024", "2025", "2026", "0.05", "0.5", "0.75",
          "1.5", "1.96", "06796", "3.11", "2.5"}

rows = []
for fn in tuple(os.environ.get("CAMSCAST_FILES",
                "body.tex,SI_serra.tex").split(",")):
    txt = open(os.path.join(ROOT, "manuscript", fn)).read()
    txt = STRIP.sub(" ", txt)
    # do not treat the decimal point inside a number as a sentence end: LaTeX math
    # such as $0.648$ would otherwise be split into a bare "648" with no context
    SENT = re.compile(r"[^!?\n]{0,400}?(?:(?<![0-9])[.](?![0-9])|[!?])")
    for m in SENT.finditer(txt):
        sent = m.group(0)
        if sent.strip().startswith("%"):
            continue
        for nm in NUM.finditer(sent):
            raw = nm.group(0).replace("{,}", "")
            if raw in IGNORE:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            hit = values.get(round(v, 4)) or values.get(round(v, 3)) or \
                  values.get(round(v, 2)) or values.get(round(v, 1)) or values.get(round(v, 0))
            rows.append(dict(file=fn, value=raw,
                             matched=bool(hit),
                             sources="; ".join(sorted(hit)[:3]) if hit else "",
                             context=re.sub(r"\s+", " ", sent.strip())[:220]))

d = pd.DataFrame(rows).drop_duplicates(subset=["file", "value", "context"])
d.to_csv(os.path.join(RUN, "number_audit.csv"), index=False)
un = d[~d.matched]
un.to_csv(os.path.join(RUN, "number_audit_unmatched.csv"), index=False)
print(f"numbers checked: {len(d)}  matched to an artefact: {int(d.matched.sum())} "
      f"({100*d.matched.mean():.1f}%)  to review: {len(un)}", flush=True)
print("\n=== review list by file ===")
print(un.groupby("file").size().to_string())
