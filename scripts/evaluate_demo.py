#!/usr/bin/env python
"""Deterministic evaluation of the match flow over a small golden set.

Every case asserts the EXPECTED deterministic eligibility status for a scheme, so a
regression in the rule engine, threshold_map behaviour or intake extraction shows up
as a failed case. The harness itself computes nothing about eligibility - it only
compares the pipeline's output to the golden expectations.

Usage:
    .venv/bin/python scripts/evaluate_demo.py            # run all cases, print report
    .venv/bin/python scripts/evaluate_demo.py --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.graph import initial_state, run_flow  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.tools.registry import build_gateway  # noqa: E402

# (case_id, query, facts, scheme_id, expected_status)
GOLDEN_SET = [
    (
        "nfsdc_rural_sc_at_boundary",
        "Rural SC student, 19 years old, undergraduate course, family income 81000 per year, needs 80000 loan",
        {},
        "demo_nfsdc_education_loan",
        "potentially_eligible",
    ),
    (
        "nfsdc_rural_sc_one_rupee_over",
        "Rural SC student, 19, undergraduate, family income is 81001 per year",
        {},
        "demo_nfsdc_education_loan",
        "not_eligible",
    ),
    (
        "nfsdc_urban_sc_under_urban_limit",
        "Urban SC student, 20 years old, undergraduate, annual income 100000",
        {},
        "demo_nfsdc_education_loan",
        "potentially_eligible",
    ),
    (
        "nbcfdc_obc_urban_eligible",
        "Urban OBC student, 20 years old, undergraduate, family income 110000 per year",
        {},
        "demo_nbcfdc_education_loan",
        "potentially_eligible",
    ),
    (
        "nbcfdc_sc_is_not_eligible",
        "Rural SC student, 20, undergraduate, income 60000",
        {},
        "demo_nbcfdc_education_loan",
        "not_eligible",
    ),
    (
        "micro_finance_woman_rural_eligible",
        "I am a 28 year old woman from a village running a tailoring shop 3 years, family income 50000 per year",
        {},
        "demo_micro_finance",
        "potentially_eligible",
    ),
    (
        "nfsdc_missing_income_is_insufficient",
        "Rural SC student, 22, undergraduate course",
        {},
        "demo_nfsdc_education_loan",
        "insufficient_data",
    ),
    (
        "nfsdc_underage_is_not_eligible",
        "Rural SC student, 17 years old, undergraduate, income 70000",
        {},
        "demo_nfsdc_education_loan",
        "not_eligible",
    ),
    (
        "structured_facts_direct",
        "",
        {"annual_income": 81000, "category": "SC", "location_type": "rural", "age": 19, "course_level": "undergraduate"},
        "demo_nfsdc_education_loan",
        "potentially_eligible",
    ),
]


async def run_eval() -> dict:
    gateway = build_gateway()
    settings = get_settings()
    results = []
    for case_id, query, facts, scheme_id, expected in GOLDEN_SET:
        state = initial_state(
            mode="match",
            request_id=f"eval-{case_id}",
            query=query,
            facts=facts or None,
            scheme_ids=[scheme_id],
            allow_external_fallback=False,  # keep the eval fully offline
            top_k=2,
        )
        final = await run_flow(gateway, state)
        evaluations = {e.scheme_id: e for e in final.get("evaluations", [])}
        evaluation = evaluations.get(scheme_id)
        actual = evaluation.eligibility.status if evaluation else "missing"
        verdicts = (final.get("verdicts_by_scheme") or {}).get(scheme_id, [])
        results.append(
            {
                "case_id": case_id,
                "scheme_id": scheme_id,
                "expected": expected,
                "actual": actual,
                "pass": actual == expected,
                "rules_evaluated": len(evaluation.eligibility.reasons) if evaluation else 0,
                "evidence_verdicts": len(verdicts),
                "warnings": len(final.get("warnings", [])),
            }
        )
    passed = sum(1 for r in results if r["pass"])
    report = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "all_passed": passed == len(results),
        "mock_mode": settings.mock_mode_flags(),
        "results": results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="NidhiSetu deterministic eval")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report = asyncio.run(run_eval())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"NidhiSetu eval: {report['passed']}/{report['total']} cases passed")
        print(f"mock_mode: {report['mock_mode']}")
        for case in report["results"]:
            mark = "PASS" if case["pass"] else "FAIL"
            print(
                f"  [{mark}] {case['case_id']:<38} expected={case['expected']:<20} "
                f"actual={case['actual']:<20} verdicts={case['evidence_verdicts']}"
            )
        if not report["all_passed"]:
            for case in report["results"]:
                if not case["pass"]:
                    print(f"MISMATCH {case['case_id']}: expected {case['expected']}, got {case['actual']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
