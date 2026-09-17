#!/usr/bin/env python3
"""Create a new Zenodo version of the prime-compiler record (v2.1 -> v3).

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
CONCEPT_RECORD = 21179525          # the v2.1 record (latest published version) to base the new version on
V3 = os.path.expanduser("~/prime-compiler/zenodo_v3")
OLD_FILES = {"prime_compiler_v2.1.pdf", "prime_compiler_v2.1.tex"}  # replaced; fig_rmsnorm_spice.png is kept in the record
NEW_FILES = [os.path.join(V3, f) for f in [
    "prime_compiler_v3.pdf", "prime_compiler_v3.tex",
    "supplementary_factorizations_v3.pdf",
    "prime_compiler_v3_code.zip",
    "fig_v3_calculus.png", "fig_v3_mamba.png", "fig_v3_noise.png",
    "fig_v3_signflip.png", "fig_v3_training.png",
]]
# Full metadata for v3 (title, description, keywords, related identifiers, version)
METADATA = json.load(open(os.path.join(V3, "zenodo_metadata_v3.json")))["metadata"]

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

# 3. upload v3 files
for path in NEW_FILES:
    name = os.path.basename(path)
    with open(path, "rb") as fh:
        req("PUT", f"{bucket}/{name}", data=fh.read(), raw=True)
    print(f"[3] uploaded {name}")

# 4. update metadata from zenodo_metadata_v3.json (keeps creators from the draft)
md = draft["metadata"]
for k in ("title", "description", "keywords", "related_identifiers", "version", "language", "license", "upload_type", "publication_type", "access_right"):
    if k in METADATA:
        md[k] = METADATA[k]
req("PUT", draft_url, data={"metadata": md})
print(f"[4] metadata updated (version={md['version']}, description/keywords/related identifiers from zenodo_metadata_v3.json)")

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
