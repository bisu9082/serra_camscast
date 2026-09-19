"""Verify every DOI in the bibliography against CrossRef.

A fabricated or mistyped DOI is the failure mode this checks for, and it is not
one a human catches by reading: the string looks like a DOI either way. Each entry
is resolved at CrossRef and the returned title, year and container are compared
with what the .bib claims, so three kinds of error surface separately:

  * unresolved   the DOI does not exist at CrossRef
  * title drift  the DOI resolves, but to a different paper than the entry names
  * metadata     the DOI and title agree, but the year or journal in the .bib is wrong

Entries with no DOI are listed rather than skipped silently, because a reference
that cannot be checked is a different state from one that passed.

Writes doi_verify.csv.
"""
import csv, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BIB = os.environ.get("CAMSCAST_BIB", os.path.join(REPO, "manuscript", "refs_ems_v1.bib"))
OUT = os.environ.get("CAMSCAST_OUT", os.path.join(REPO, "results", "doi_verify.csv"))
MAILTO = "bisu9082@gmail.com"          # CrossRef asks callers to identify themselves
PAUSE = float(os.environ.get("PAUSE", 0.25))


def entries(path):
    txt = open(path).read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", txt, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        def field(name):
            # the final field of an entry carries no trailing newline inside the body,
            # so the line terminator has to be optional or that field is silently missed
            f = re.search(name + r"\s*=\s*[{\"](.+?)[}\"]\s*,?\s*(?:\n|$)", body, re.S | re.I)
            return re.sub(r"\s+", " ", f.group(1)).strip() if f else ""
        out.append(dict(key=key, kind=kind, title=field("title"), year=field("year"),
                        doi=field("doi"), journal=field("journal") or field("booktitle")))
    return out


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def overlap(a, b):
    """Fraction of the shorter title's words that appear in the other."""
    wa, wb = set(norm(a).split()), set(norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _get(url, tries=3):
    """Transient network failures are not evidence about a DOI, so retry before
    concluding anything."""
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": f"camscast-audit (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise                                    # a 404 is an answer, not a failure
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def datacite(doi):
    """arXiv and Zenodo DOIs (10.48550/, 10.5281/) are registered with DataCite, not
    CrossRef, so a CrossRef 404 for them means 'wrong registry', not 'fabricated'."""
    m = _get("https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe=""))["data"]["attributes"]
    titles = m.get("titles") or [{}]
    return dict(title=titles[0].get("title", ""),
                year=str(m.get("publicationYear") or ""),
                container=(m.get("publisher") or ""),
                type=(m.get("types") or {}).get("resourceTypeGeneral", ""),
                registry="DataCite")


def crossref(doi):
    try:
        m = _get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return datacite(doi)                     # try the other registry before failing
        raise
    yr = ""
    for k in ("published-print", "published-online", "issued", "created"):
        if m.get(k, {}).get("date-parts", [[None]])[0][0]:
            yr = str(m[k]["date-parts"][0][0]); break
    return dict(title=(m.get("title") or [""])[0],
                year=yr,
                container=(m.get("container-title") or [""])[0],
                type=m.get("type", ""), registry="CrossRef")


rows, tally = [], {"ok": 0, "title_drift": 0, "metadata": 0, "unresolved": 0, "no_doi": 0}
E = entries(BIB)
print(f"bibliography: {len(E)} entries", flush=True)

for i, e in enumerate(E, 1):
    if not e["doi"]:
        tally["no_doi"] += 1
        rows.append(dict(status="no_doi", **e, cr_title="", cr_year="", cr_container="",
                         title_overlap=""))
        print(f"[{i:3d}/{len(E)}] {e['key']:<24} NO DOI  {e['title'][:60]}", flush=True)
        continue
    try:
        cr = crossref(e["doi"])
    except Exception as ex:
        tally["unresolved"] += 1
        rows.append(dict(status="unresolved", **e, cr_title=f"ERROR {ex}",
                         cr_year="", cr_container="", title_overlap=""))
        print(f"[{i:3d}/{len(E)}] {e['key']:<24} UNRESOLVED  {e['doi']}  ({ex})", flush=True)
        time.sleep(PAUSE)
        continue
    ov = overlap(e["title"], cr["title"])
    if ov < 0.6:
        st = "title_drift"
    elif e["year"] and cr["year"] and e["year"] != cr["year"]:
        st = "metadata"
    else:
        st = "ok"
    tally[st] += 1
    rows.append(dict(status=st, **e, cr_title=cr["title"], cr_year=cr["year"],
                     cr_container=cr["container"] + f" [{cr.get('registry','')}]",
                     title_overlap=round(ov, 2)))
    flag = "" if st == "ok" else f"  <-- {st.upper()}"
    print(f"[{i:3d}/{len(E)}] {e['key']:<24} {st:<12} ov={ov:.2f}{flag}", flush=True)
    time.sleep(PAUSE)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["status", "key", "kind", "title", "year", "doi",
                                      "journal", "cr_title", "cr_year", "cr_container",
                                      "title_overlap"])
    w.writeheader(); w.writerows(rows)

print("\n=== summary ===")
for k in ("ok", "metadata", "title_drift", "unresolved", "no_doi"):
    print(f"  {k:<12} {tally[k]}")
bad = tally["title_drift"] + tally["unresolved"]
print(f"\nfabricated or wrong DOIs: {bad}")
print(f"saved {os.path.basename(OUT)}")
sys.exit(1 if bad else 0)
