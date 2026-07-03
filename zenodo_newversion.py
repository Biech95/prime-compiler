#!/usr/bin/env python3
"""Create a new Zenodo version of the prime-compiler record (v2 -> v2.1).

Token is read from the environment ($ZENODO_TOKEN) and never written to disk.

Usage:
    export ZENODO_TOKEN=...          # your personal Zenodo token
    python3 zenodo_newversion.py     # prepares a DRAFT, stops before publishing
    python3 zenodo_newversion.py --publish   # same, then publishes (irreversible!)

The default run is fully reversible: it creates a draft you can review or
discard on zenodo.org. Only --publish mints the new version DOI.
"""
import os
import sys
import json
import urllib.request

API = "https://zenodo.org/api"
CONCEPT_RECORD = 21138358          # the published record to base the new version on
PDF = os.path.expanduser("~/Schreibtisch/prime_compiler_v2.1.pdf")
TEX = os.path.expanduser("~/Schreibtisch/prime_compiler_v2.1.tex")
OLD_FILES = {"prime_compiler_v2.pdf", "prime_compiler_v2.tex"}  # replaced; fig kept
NEW_FILES = [PDF, TEX]
CHANGELOG = (
    "v2.1: Retrospective chip analysis (Sec. 6) reframed from verdict to finding "
    "-- no chip team's decision is called 'suboptimal'; the section now states "
    "additional analog mapping opportunities and their platform prerequisites, and "
    "notes that the circuit-level results justify the teams' caution toward open-loop "
    "designs. Table 8 gains per-row feasibility markers (standard-CMOS vs. device-"
    "technology prerequisite) and a scope caveat (energy-only). Editorial pass: "
    "corrected the Sillman (2023) citation title; reconciled the transformer-layer "
    "transition count with its diagram; referenced the fusion table; removed two "
    "unused theorem environments; separated the CN101 chip from its Nature Comms "
    "system paper."
)

TOKEN = os.environ.get("ZENODO_TOKEN")
if not TOKEN:
    sys.exit("ERROR: set ZENODO_TOKEN in the environment first.")
DO_PUBLISH = "--publish" in sys.argv
H = {"Authorization": f"Bearer {TOKEN}"}


def req(method, url, data=None, headers=None, raw=False):
    hdrs = dict(H)
    if headers:
        hdrs.update(headers)
    body = None
    if data is not None and not raw:
        body = json.dumps(data).encode()
        hdrs["Content-Type"] = "application/json"
    elif raw:
        body = data
        hdrs["Content-Type"] = "application/octet-stream"
    r = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    with urllib.request.urlopen(r) as resp:
        txt = resp.read().decode()
        return resp.status, (json.loads(txt) if txt else {})


# 1. new version draft (or resume an existing one via --draft <id>)
resume = None
for i, a in enumerate(sys.argv):
    if a == "--draft" and i + 1 < len(sys.argv):
        resume = sys.argv[i + 1]
if resume:
    draft_url = f"{API}/deposit/depositions/{resume}"
    st, draft = req("GET", draft_url)
    did = draft["id"]
    print(f"[1] resuming draft id={did}  state={draft.get('state')}")
else:
    st, dep = req("POST", f"{API}/deposit/depositions/{CONCEPT_RECORD}/actions/newversion")
    draft_url = dep["links"]["latest_draft"]
    st, draft = req("GET", draft_url)
    did = draft["id"]
    print(f"[1] new draft id={did}  state={draft.get('state')}")

# 2. delete the old v2 files, keep the figure
bucket = draft["links"]["bucket"]
for f in draft.get("files", []):
    if f["filename"] in OLD_FILES:
        req("DELETE", f["links"]["self"])
        print(f"[2] deleted inherited {f['filename']}")

# 3. upload v2.1 files
for path in NEW_FILES:
    name = os.path.basename(path)
    with open(path, "rb") as fh:
        req("PUT", f"{bucket}/{name}", data=fh.read(), raw=True)
    print(f"[3] uploaded {name}")

# 4. update metadata (version + append changelog to description)
md = draft["metadata"]
md["version"] = "v2.1"
md["description"] = md.get("description", "") + f"<p><strong>Changelog {md['version']}:</strong> {CHANGELOG}</p>"
req("PUT", draft_url, data={"metadata": md})
print("[4] metadata updated (version=v2.1, changelog appended)")

print(f"\nDRAFT READY: https://zenodo.org/deposit/{did}")
print("Review the draft in your browser. It is not yet public.")

# 5. publish (only with --publish)
if DO_PUBLISH:
    st, pub = req("POST", f"{API}/deposit/depositions/{did}/actions/publish")
    print(f"\n[5] PUBLISHED. New version DOI: {pub.get('doi')}")
    print(f"    Record: {pub['links'].get('record_html')}")
else:
    print("\nNot published (reversible). To mint the new-version DOI, either:")
    print("  - click Publish on the draft page above, or")
    print("  - re-run:  python3 zenodo_newversion.py --publish")
