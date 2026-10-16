"""Download the FastAPI tutorial pages used as the demo corpus into data/corpus/."""
import sys
from pathlib import Path

import httpx

BASE = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial"
PAGES = {
    "fastapi_first_steps.md": "first-steps.md",
    "fastapi_path_params.md": "path-params.md",
    "fastapi_query_params.md": "query-params.md",
    "fastapi_body.md": "body.md",
    "fastapi_dependencies_index.md": "dependencies/index.md",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "corpus"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for local, remote in PAGES.items():
        r = httpx.get(f"{BASE}/{remote}", timeout=30, follow_redirects=True)
        if r.status_code != 200:
            sys.exit(f"Failed to fetch {remote}: HTTP {r.status_code}")
        (OUT / local).write_text(r.text, encoding="utf-8")
        print(f"saved {local} ({len(r.text)} chars)")
