"""AI-style detector for the manuscript prose.

Flags the constructions that mark machine-generated academic English, with the
sentence each one sits in so a human decision can be made. It changes nothing on
its own: it writes a review list.

The categories are the ones that actually discriminate, not a generic style guide:

  1. Inflated connectives and hedges that carry no information
     ("moreover", "furthermore", "it is important to note", "notably").
  2. The triad habit --- three parallel adjectives or noun phrases in a row, which
     LLMs produce far more often than human writers do.
  3. "Not only X but also Y" and "It is X that Y" cleft frames.
  4. Nominalisation stacks ("the utilization of", "in order to", "a variety of").
  5. Self-congratulatory evaluation of one's own work ("robust", "comprehensive",
     "novel", "significant contribution", "sheds light on", "paves the way").
  6. Unmeasurable qualifiers, which the project's own style rules forbid
     ("appropriately", "adequately", "sufficiently", "generally", "carefully").
  7. Em-dash density and sentence-length uniformity, reported as file statistics
     rather than per sentence.

Writes ai_review.csv and prints a per-file summary.
"""
import csv, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.environ.get("CAMSCAST_TEX", os.path.join(REPO, "manuscript"))
FILES = tuple(os.environ.get("CAMSCAST_FILES",
               "body.tex,main_serra_sn.tex,SI_serra.tex").split(","))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(REPO, "results", "ai_review.csv"))

PATTERNS = [
    ("connective", r"\b(?:Moreover|Furthermore|Additionally|In addition|Notably|Importantly|Indeed|Overall|In conclusion|To summari[sz]e|That said|Nevertheless|Nonetheless)\b"),
    ("meta_hedge", r"\b(?:[Ii]t is (?:important|worth|crucial|essential|interesting) to (?:note|mention|highlight|emphasi[sz]e)|[Ii]t should be noted)\b"),
    ("cleft", r"\b(?:[Nn]ot only .{0,60}? but also|[Ii]t is .{0,40}? that\b)"),
    ("nominalisation", r"\b(?:the utili[sz]ation of|in order to|a variety of|a range of|a myriad of|serves? as a|plays? a (?:key|crucial|vital|significant) role|the fact that)\b"),
    ("self_praise", r"\b(?:robustly|comprehensive|novel approach|significant contribution|sheds? light on|paves? the way|underscores?|highlights? the importance|valuable insights?|state[- ]of[- ]the[- ]art)\b"),
    ("unmeasurable", r"\b(?:appropriately|adequately|sufficiently|generally speaking|carefully|properly|effectively|successfully|significantly improve)\b"),
    ("delve", r"\b(?:delve|leverage|harness|realm|landscape|tapestry|intricate|pivotal|multifaceted|holistic|paradigm shift)\b"),
]

STRIP = re.compile(r"\\(?:label|ref|cite[tp]?|documentclass|usepackage|input|include|begin|end)\{[^}]*\}")
TRIAD = re.compile(r"\b(\w+ly|\w+ent|\w+ive|\w+al)\b,\s+\b\w+(?:ly|ent|ive|al)\b,?\s+and\s+\b\w+(?:ly|ent|ive|al)\b")

rows = []
summary = {}
for fn in FILES:
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p):
        print("missing", p); continue
    txt = STRIP.sub(" ", open(p).read())
    # drop comment lines and tabular bodies, which are not prose
    txt = "\n".join(L for L in txt.split("\n") if not L.strip().startswith("%"))
    txt = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", txt, flags=re.S)
    sents = [s.strip() for s in re.findall(r"[^.!?\n]{20,600}[.!?]", txt)]
    counts = {}
    for s in sents:
        for name, pat in PATTERNS:
            for m in re.finditer(pat, s):
                counts[name] = counts.get(name, 0) + 1
                rows.append(dict(file=fn, category=name, hit=m.group(0),
                                 sentence=re.sub(r"\s+", " ", s)[:300]))
        for m in TRIAD.finditer(s):
            counts["triad"] = counts.get("triad", 0) + 1
            rows.append(dict(file=fn, category="triad", hit=m.group(0),
                             sentence=re.sub(r"\s+", " ", s)[:300]))
    lens = np.array([len(s.split()) for s in sents]) if sents else np.array([0])
    summary[fn] = dict(sentences=len(sents),
                       words=int(lens.sum()),
                       mean_len=float(lens.mean()),
                       sd_len=float(lens.std(ddof=1)) if len(lens) > 1 else 0.0,
                       cv_len=float(lens.std(ddof=1) / lens.mean()) if len(lens) > 1 and lens.mean() else 0.0,
                       em_dash_per_1000w=float(1000 * txt.count("---") / max(lens.sum(), 1)),
                       flags=sum(counts.values()), by_category=counts)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "category", "hit", "sentence"])
    w.writeheader(); w.writerows(rows)

print(f"{'file':<22}{'sents':>7}{'words':>8}{'mean':>7}{'sd':>7}{'CV':>7}{'em/1kw':>8}{'flags':>7}")
for fn, s in summary.items():
    print(f"{fn:<22}{s['sentences']:>7}{s['words']:>8}{s['mean_len']:>7.1f}{s['sd_len']:>7.1f}"
          f"{s['cv_len']:>7.2f}{s['em_dash_per_1000w']:>8.1f}{s['flags']:>7}")
print("\nby category:")
allc = {}
for s in summary.values():
    for k, v in s["by_category"].items():
        allc[k] = allc.get(k, 0) + v
for k in sorted(allc, key=lambda x: -allc[x]):
    print(f"  {k:<16} {allc[k]}")
print(f"\nreview list: {OUT}  ({len(rows)} rows)")
print("reference: human-written academic prose runs CV of sentence length 0.45-0.70; "
      "uniform machine prose sits below 0.40.")
