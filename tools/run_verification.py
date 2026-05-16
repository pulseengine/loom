#!/usr/bin/env python3
"""Execute every rivet `type: feature` artifact's `fields.steps[].run` commands.

Reads artifacts via `rivet list --filter <sexp>` + `rivet get <id> --format json`,
runs each step under a configurable shell, collects per-artifact pass/fail, and
writes a structured JSON summary alongside human-readable stdout progress.

Usage:
    tools/run_verification.py [--filter '<sexp>'] [--results-json PATH] [--shell SHELL]

Defaults:
    --filter '(= type "feature")'
    --results-json verification-results.json
    --shell        bash

Exit code:
    0  if every matched artifact's steps passed (or were skipped)
    1  if any artifact failed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Result:
    filter: str
    total: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def rivet_list_ids(filter_sexp: str) -> list[str]:
    proc = subprocess.run(
        ["rivet", "list", "--filter", filter_sexp, "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    return [a["id"] for a in data.get("artifacts", [])]


def rivet_get_steps(artifact_id: str) -> list[str]:
    proc = subprocess.run(
        ["rivet", "get", artifact_id, "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    return [s["run"] for s in data.get("fields", {}).get("steps", []) if "run" in s]


def run_one_step(cmd: str, shell: str) -> bool:
    """Return True iff exit code is 0."""
    proc = subprocess.run(
        [shell, "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter",
        default='(= type "feature")',
        help='rivet s-expression filter (default: %(default)r)',
    )
    parser.add_argument(
        "--results-json",
        default="verification-results.json",
        type=Path,
        help="path for the JSON summary (default: %(default)s)",
    )
    parser.add_argument(
        "--shell",
        default=os.environ.get("VERIFY_SHELL", "bash"),
        help="shell used to execute each step (default: %(default)s)",
    )
    args = parser.parse_args()

    result = Result(filter=args.filter)

    print("== rivet verification gate ==")
    print(f"filter: {args.filter}")
    print(f"shell:  {args.shell}")
    print()

    try:
        ids = rivet_list_ids(args.filter)
    except subprocess.CalledProcessError as e:
        print(f"rivet list failed: {e.stderr}", file=sys.stderr)
        return 2

    if not ids:
        print("No artifacts matched filter.", file=sys.stderr)
        args.results_json.write_text(json.dumps(asdict(result), indent=2))
        return 1

    print(f"matched {len(ids)} artifacts")
    print()
    result.total = len(ids)

    for artifact_id in ids:
        try:
            steps = rivet_get_steps(artifact_id)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {artifact_id}: rivet get failed: {e.stderr}")
            result.failed.append(artifact_id)
            continue

        if not steps:
            print(f"[SKIP] {artifact_id} (no fields.steps[].run)")
            result.skipped.append(artifact_id)
            continue

        print(f"[RUN ] {artifact_id}")
        ok = True
        for cmd in steps:
            print(f"       + {cmd}")
            if not run_one_step(cmd, args.shell):
                ok = False
                print(f"       ✗ failed: {cmd}")
                break

        if ok:
            print(f"[ OK ] {artifact_id}")
            result.passed.append(artifact_id)
        else:
            print(f"[FAIL] {artifact_id}")
            result.failed.append(artifact_id)

    result.passed_count = len(result.passed)
    result.failed_count = len(result.failed)
    result.skipped_count = len(result.skipped)

    args.results_json.write_text(json.dumps(asdict(result), indent=2))

    print()
    print("== summary ==")
    print(f"passed:  {result.passed_count}")
    print(f"failed:  {result.failed_count}")
    print(f"skipped: {result.skipped_count}")
    if result.failed:
        print("failed IDs:")
        for fid in result.failed:
            print(f"  - {fid}")

    return 0 if result.failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
