"""Re-check every dataset and model link in docs/datasets.md.

Links rot, and a dead link in a paper's data-availability statement is a reviewer's first
easy objection. Run this before any submission.

Usage:  python whose-eyes/tools/check_links.py
"""

from __future__ import annotations

import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(HERE), "docs", "datasets.md")

# A generic client name: the user's identity never goes to an external service.
HEADERS = {"User-Agent": "research-link-checker/1.0"}
TIMEOUT = 20


#: Known-good responses that are not HTTP 200.
#:   figshare answers 202 Accepted for dataset landing pages.
#:   Deep Blue (RIGA) answers 403 to any non-browser client; the page works when opened by hand.
ACCEPTABLE = {200, 202}
BROWSER_ONLY = ("deepblue.lib.umich.edu",)


def check(url: str) -> tuple[int | str, str]:
    request = urllib.request.Request(url, headers=HEADERS, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.geturl()
    except urllib.error.HTTPError as error:
        return error.code, url
    except Exception as error:  # noqa: BLE001 - network failures are the point here
        return type(error).__name__, url


def main() -> int:
    with open(DOC, encoding="utf-8") as fh:
        text = fh.read()
    urls = sorted(set(re.findall(r"<(https?://[^>]+)>", text)))
    print(f"{len(urls)} links in docs/datasets.md\n")
    bad = 0
    for url in urls:
        status, final = check(url)
        browser_only = any(host in url for host in BROWSER_ONLY)
        ok = status in ACCEPTABLE or (browser_only and status == 403)
        bad += not ok
        note = "" if final == url else f"  -> {final}"
        if browser_only and status == 403:
            note += "   (403 to scripts by design; opens fine in a browser)"
        print(f"  {'OK ' if ok else 'BAD'}  {str(status):<12} {url}{note}")
    print(f"\n{bad} link(s) need attention." if bad else "\nAll links resolved.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
