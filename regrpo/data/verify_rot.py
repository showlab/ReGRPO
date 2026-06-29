# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Audit the quality and validity of a generated RoT corpus.

Runs the strict record validator plus independent structural, grounding,
label-leak, and SFT-loadability checks, and prints a summary report. Exit code
is non-zero if any hard check fails, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Two structural "Code:" markers separated only by whitespace right before the
# fence (the reconstruction-duplication bug). NOT prose that merely says "Code:".
DOUBLED_MARKER_RE = re.compile(r"Code\s*:\s*\n\s*Code\s*:")

from regrpo.common.io import read_json
from regrpo.common.schema import (
    _final_response_index,
    evidence_grounded,
    has_label_leak,
    validate_rot_record,
)
from regrpo.common.trajectory import CODE_BLOCK_RE, extract_tool_calls


def _code_of(content: str) -> str:
    match = CODE_BLOCK_RE.search(content)
    return match.group(1).strip() if match else ""


def _reflection_turn(conversations: list[dict]) -> dict | None:
    for turn in conversations:
        if turn.get("role") == "assistant" and str(turn.get("content", "")).startswith("Reflection:"):
            return turn
    return None


def audit(records: list[dict]) -> dict[str, Any]:
    """Return a structured audit of the corpus."""

    report: dict[str, Any] = {
        "total": len(records),
        "schema_invalid": 0,
        "duplicate_ids": 0,
        "ungrounded_evidence": 0,
        "label_leaks": 0,
        "empty_corrected_code": 0,
        "doubled_code_marker": 0,
        "failed_equals_corrected": 0,
        "reflection_turn_missing": 0,
        "rot_mask_empty": 0,
        "error_type_dist": Counter(),
        "primary_tool_dist": Counter(),
        "examples": [],
    }

    seen_ids: set[str] = set()
    for rec in records:
        meta = rec.get("rot_meta", {})
        report["error_type_dist"][meta.get("error_type")] += 1

        rec_id = str(rec.get("id"))
        if rec_id in seen_ids:
            report["duplicate_ids"] += 1
        seen_ids.add(rec_id)

        if validate_rot_record(rec):
            report["schema_invalid"] += 1

        evidence = str(meta.get("reflection", {}).get("evidence", ""))
        failed_obs = str(meta.get("failed_observation", ""))
        if not evidence_grounded(evidence, failed_obs):
            report["ungrounded_evidence"] += 1

        corrected = str(meta.get("corrected_action", ""))
        if has_label_leak(rec.get("conversations", []), corrected):
            report["label_leaks"] += 1

        corrected_code = _code_of(corrected)
        if not corrected_code or not extract_tool_calls(corrected_code):
            report["empty_corrected_code"] += 1
        if DOUBLED_MARKER_RE.search(corrected):
            report["doubled_code_marker"] += 1

        failed_code = _code_of(str(meta.get("failed_action", "")))
        if failed_code and failed_code == corrected_code:
            report["failed_equals_corrected"] += 1

        # primary tool of the corrected (true) action
        calls = extract_tool_calls(corrected_code)
        non_final = [c.name for c in calls if c.name != "final_answer"]
        report["primary_tool_dist"][non_final[-1] if non_final else "final_answer"] += 1

        conversations = rec.get("conversations", [])
        if _reflection_turn(conversations) is None:
            report["reflection_turn_missing"] += 1

        # RoT loss mask must be able to locate the reflection target turn
        if _final_response_index(conversations) < 0:
            report["rot_mask_empty"] += 1

    report["error_type_dist"] = dict(report["error_type_dist"])
    report["primary_tool_dist"] = dict(report["primary_tool_dist"].most_common())
    report["examples"] = [r["id"] for r in records[:3]]
    return report


HARD_CHECKS = (
    "schema_invalid",
    "duplicate_ids",
    "ungrounded_evidence",
    "label_leaks",
    "empty_corrected_code",
    "doubled_code_marker",
    "failed_equals_corrected",
    "reflection_turn_missing",
    "rot_mask_empty",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="dataset/rot_train.json")
    args = parser.parse_args()

    records = read_json(args.path)
    if not isinstance(records, list):
        raise ValueError(f"{args.path} must be a JSON array")
    report = audit(records)

    print(f"== RoT corpus audit: {args.path} ==")
    print(f"total records:            {report['total']}")
    print(f"error_type distribution:  {report['error_type_dist']}")
    print(f"corrected primary tool:   {report['primary_tool_dist']}")
    print("-- hard checks (all must be 0) --")
    failures = 0
    for key in HARD_CHECKS:
        flag = "" if report[key] == 0 else "  <-- FAIL"
        if report[key]:
            failures += report[key]
        print(f"  {key:24s} {report[key]}{flag}")
    valid = report["total"] - report["schema_invalid"]
    print(f"-- {valid}/{report['total']} records pass every check --")
    if failures:
        print(f"AUDIT FAILED: {failures} problem(s) found")
        sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
