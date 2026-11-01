"""Re-download the PEP corpus and check it against data/peps/MANIFEST.json.

The corpus is committed, so you don't need this to run anything. It exists so
anyone can see where every file came from and confirm nothing was edited.

    python scripts/fetch_corpus.py          # verify the committed files
    python scripts/fetch_corpus.py --fetch  # download fresh copies, then compare
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "peps"
MANIFEST = DATA / "MANIFEST.json"
URL = "https://raw.githubusercontent.com/python/peps/main/peps/{name}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="download upstream copies")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    changed = 0
    for name, expected in manifest["files"].items():
        if args.fetch:
            with urllib.request.urlopen(URL.format(name=name), timeout=30) as resp:
                data = resp.read()
            label = "upstream"
        else:
            data = (DATA / name).read_bytes()
            label = "local"
        ok = sha256(data) == expected
        changed += not ok
        print(f"{'ok     ' if ok else 'CHANGED'} {label} {name}")
    if changed and args.fetch:
        print(f"\n{changed} file(s) differ upstream: the PEPs were edited after the fetch date.")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
