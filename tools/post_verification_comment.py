#!/usr/bin/env python3
"""Post (or update) a sticky PR comment summarising rivet verification results.

Reads the JSON written by `tools/run_verification.py` and calls the GitHub
REST API directly to upsert a single marker-tagged comment on the PR.
Re-running on the same PR replaces the prior body rather than appending
another comment. Pure stdlib (urllib) — no `gh` CLI dependency.

Usage:
    tools/post_verification_comment.py <pr-number> [--results-json PATH] [--repo OWNER/NAME]

Required env:
    GH_TOKEN (or GITHUB_TOKEN) with `issues: write` (PR comments go through
        the ISSUES API even on a pull request).
    GH_REPO (or GITHUB_REPOSITORY) naming the target OWNER/NAME, unless
        --repo is passed. There is no default: see #420.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MARKER = "<!-- rivet-verification-gate -->"
API = "https://api.github.com"


def github_request(
    method: str, path: str, token: str, body: dict | None = None
) -> tuple[int, bytes]:
    url = f"{API}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rivet-verification-gate",
            "Content-Type": "application/json" if data else "application/octet-stream",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def render_body(results: dict) -> str:
    passed = results["passed_count"]
    failed = results["failed_count"]
    skipped = results["skipped_count"]
    total = results["total"]
    failed_ids = results["failed"]
    flt = results["filter"]

    if failed == 0:
        status = f"✅ **{passed}/{total}** passed"
    else:
        status = f"❌ **{passed}/{total}** passed — **{failed}** failed"

    failed_section = (
        "\n".join(f"- `{i}`" for i in failed_ids) if failed_ids else "_(none)_"
    )

    return f"""{MARKER}
## Rivet verification gate

{status}

| | count |
|---|---:|
| Passed  | {passed} |
| Failed  | {failed} |
| Skipped (no steps) | {skipped} |

**Filter:** `{flt}`

<details><summary>Failed artifacts</summary>

{failed_section}

</details>

<sub>Updated automatically by `tools/post_verification_comment.py`. Source of truth: `artifacts/verification.yaml`.</sub>"""


def find_marker_comment(repo: str, pr: int, token: str) -> int | None:
    """Page through PR comments looking for the marker. Returns comment id or None."""
    page = 1
    while True:
        status, body = github_request(
            "GET",
            f"/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}",
            token,
        )
        if status != 200:
            print(f"GET comments failed: {status} {body[:200]}", file=sys.stderr)
            return None
        comments = json.loads(body)
        if not comments:
            return None
        for c in comments:
            if MARKER in (c.get("body") or ""):
                return c["id"]
        if len(comments) < 100:
            return None
        page += 1


def upsert_comment(repo: str, pr: int, body: str, token: str) -> None:
    existing = find_marker_comment(repo, pr, token)
    if existing is not None:
        print(f"updating comment {existing}", file=sys.stderr)
        status, resp = github_request(
            "PATCH",
            f"/repos/{repo}/issues/comments/{existing}",
            token,
            {"body": body},
        )
    else:
        print("creating new comment", file=sys.stderr)
        status, resp = github_request(
            "POST",
            f"/repos/{repo}/issues/{pr}/comments",
            token,
            {"body": body},
        )
    if status not in (200, 201):
        print(f"comment upsert failed: {status} {resp[:300]}", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int, help="pull-request number")
    parser.add_argument(
        "--results-json",
        default="verification-results.json",
        type=Path,
        help="path to the JSON summary (default: %(default)s)",
    )
    parser.add_argument(
        "--repo",
        # #420: this used to default to the repo this script was COPIED FROM.
        # Nothing set GH_REPO here, so every run posted at that project's
        # issue of the same number and was refused 403 by a repo-scoped
        # token. A verification tool that silently picks the wrong target
        # instead of refusing is the defect class this gate exists to catch,
        # so there is deliberately no fallback now: resolve from the
        # environment Actions already provides, or fail.
        default=os.environ.get("GH_REPO") or os.environ.get("GITHUB_REPOSITORY"),
        help="OWNER/NAME (default: $GH_REPO, else $GITHUB_REPOSITORY)",
    )
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN or GITHUB_TOKEN required", file=sys.stderr)
        return 2

    if not args.repo:
        print(
            "target repository unknown: pass --repo OWNER/NAME, or set GH_REPO "
            "or GITHUB_REPOSITORY (#420 — refusing to guess)",
            file=sys.stderr,
        )
        return 2

    if not args.results_json.is_file():
        print(f"no {args.results_json} found; nothing to post", file=sys.stderr)
        return 0

    results = json.loads(args.results_json.read_text())
    body = render_body(results)
    upsert_comment(args.repo, args.pr, body, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
