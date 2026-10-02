"""Check that hard-coded table and figure numbers point at the table or figure meant.

The supplement numbers its tables by LaTeX counter but refers to them by literal text
("Table~S24"), and the main text does the same for both its own figures and the
supplement's tables. Nothing links the two, so inserting or deleting one float silently
renumbers every reference after it. That has happened twice in this manuscript's history,
once shipping two figures under each other's captions.

This script rebuilds the numbering from the source, then reports two things a person can
act on: which numbers are referenced but do not exist, and what the caption of each
referenced float actually says, so the reference can be read against its target.

Writes xref_audit.csv.
"""
import csv, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.environ.get("CAMSCAST_TEX", os.path.join(REPO, "manuscript"))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(REPO, "results", "xref_audit.csv"))

FLOAT = re.compile(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", re.S)
BOLDCAP = re.compile(r"\\caption\{\s*\\textbf\{(.+?)\}", re.S)
ANYCAP = re.compile(r"\\caption\{(.{0,120})", re.S)


def floats_in(path, prefix):
    """Number the floats in document order, the way LaTeX will."""
    txt = open(path).read()
    out = {"table": {}, "figure": {}}
    n = {"table": 0, "figure": 0}
    for m in FLOAT.finditer(txt):
        kind = m.group(1)
        n[kind] += 1
        body = m.group(0)
        c = BOLDCAP.search(body) or ANYCAP.search(body)
        cap = re.sub(r"\s+", " ", c.group(1)).strip() if c else "(no caption)"
        img = re.search(r"includegraphics\[[^\]]*\]\{([^}]*)\}", body)
        out[kind][f"{prefix}{n[kind]}"] = dict(caption=cap[:150],
                                               file=img.group(1) if img else "")
    return out


si = floats_in(os.path.join(ROOT, os.environ.get("CAMSCAST_SI", "SI_serra.tex")), "S")
main = floats_in(os.path.join(ROOT, os.environ.get("CAMSCAST_MAIN", "body.tex")), "")

known_tables = dict(si["table"])
known_tables.update(main["table"])
known_figs = dict(si["figure"])
known_figs.update(main["figure"])

print(f"supplement: {len(si['table'])} tables (S1-S{len(si['table'])}), "
      f"{len(si['figure'])} figures")
print(f"main text : {len(main['table'])} tables, {len(main['figure'])} figures")

REF_T = re.compile(r"Table~(S?\d+)")
REF_F = re.compile(r"Figure~(S?\d+)")

rows, missing = [], []
for fn in (os.environ.get("CAMSCAST_MAIN", "body.tex"),
           os.environ.get("CAMSCAST_SI", "SI_serra.tex")):
    path = os.path.join(ROOT, fn)
    if not os.path.exists(path):
        continue
    txt = open(path).read()
    for rx, kind, known in ((REF_T, "table", known_tables), (REF_F, "figure", known_figs)):
        for m in rx.finditer(txt):
            num = m.group(1)
            tgt = known.get(num)
            rows.append(dict(source=fn, kind=kind, reference=f"{kind} {num}",
                             exists=bool(tgt),
                             target_caption=tgt["caption"] if tgt else "",
                             target_file=tgt["file"] if tgt else ""))
            if not tgt:
                missing.append((fn, kind, num))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["source", "kind", "reference", "exists",
                                      "target_caption", "target_file"])
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["kind"], r["source"], r["reference"])))

uniq = {(r["kind"], r["reference"]) for r in rows}
print(f"\nreferences checked: {len(rows)} ({len(uniq)} distinct)")
if missing:
    print(f"DANGLING ({len(missing)}): references to a float that does not exist")
    for fn, kind, num in sorted(set(missing)):
        print(f"  {fn}: {kind} {num}")
else:
    print("dangling references: none")

print("\nfigures, in document order, with the image each caption is attached to:")
for k, v in main["figure"].items():
    print(f"  Figure {k or '?':<3} {v['file']:<12} {v['caption'][:78]}")
for k, v in si["figure"].items():
    print(f"  Figure {k:<3} {v['file']:<12} {v['caption'][:78]}")
print(f"\nsaved {os.path.basename(OUT)} — read it to check each reference against its target")
