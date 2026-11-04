"""Regression tests for the note writer's medication and HPI safety checks.

Run from the repo root:  python -m pytest tests/ -q
Each case is a real failure pattern: a medication start or refill that must not reach the note without support.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_note import _checks, _ground_sections  # noqa: E402

CHART = {"sex": "M", "meds": ["Lisinopril 10 mg daily", "Oxycodone 5 mg q6h"]}
LONG_VISIT = "Patient came in for knee pain. " * 20  # long enough that the HPI checks run


def _ground(plan, transcript, chart=CHART):
    sec = {"plan": plan, "assessment": "", "meds": "", "hpi": ""}
    items = []
    suggestions = _ground_sections(sec, items, transcript, chart)
    return sec["plan"], suggestions


def test_discussed_but_not_started_drug_goes_to_suggestions():
    # provider asked a question; the model wrote it as an order
    plan, sugg = _ground("Start lisinopril 10 mg daily",
                         "Should we start lisinopril? Let me think about it. " + LONG_VISIT)
    assert "Start lisinopril" not in plan
    assert "Start lisinopril 10 mg daily" in sugg


def test_affirmative_start_is_kept():
    plan, sugg = _ground("Start lisinopril 10 mg daily",
                         "I'll start you on lisinopril 10 mg daily. " + LONG_VISIT)
    assert "Start lisinopril 10 mg daily" in plan
    assert sugg == []


def test_new_drug_not_on_chart_is_never_written_as_a_change():
    plan, sugg = _ground("Start metformin 500 mg twice daily",
                         "I'll start you on metformin 500 mg twice daily. " + LONG_VISIT)
    assert "metformin" not in plan.lower()
    assert sugg


def test_refill_request_on_chart_drug_stays_in_plan():
    plan, sugg = _ground("Refill requested: Lisinopril 10 mg daily",
                         "I'm running low on my lisinopril, can you refill it? " + LONG_VISIT)
    assert "Refill requested" in plan
    assert sugg == []


def test_continue_stays_in_plan():
    plan, _ = _ground("Continue lisinopril 10 mg daily",
                      "We'll keep you on the lisinopril. " + LONG_VISIT)
    assert "Continue lisinopril" in plan


def test_pdmp_flag_tied_to_the_drug_named_near_refill_talk():
    out = {"sections": {"plan": "", "assessment": "", "meds": "", "hpi": "x" * 50}, "assessmentItems": [], "flags": []}
    flags = _checks(out, CHART, "Oxycodone is what I take. I'm running low on it. " + LONG_VISIT)
    assert any(f["title"] == "PDMP" and "Oxycodone" in f["text"] for f in flags)


def test_pdmp_not_raised_when_refill_talk_is_about_another_drug():
    out = {"sections": {"plan": "", "assessment": "", "meds": "", "hpi": "x" * 50}, "assessmentItems": [], "flags": []}
    flags = _checks(out, CHART, "I'm running low on my lisinopril. Oxycodone is fine. " + LONG_VISIT)
    assert not any(f["title"] == "PDMP" for f in flags)


def test_hpi_missing_duration_and_severity_is_flagged():
    out = {"sections": {"plan": "", "assessment": "", "meds": "", "hpi": "Knee pain, right side, worse with stairs."},
           "assessmentItems": [], "flags": []}
    flags = _checks(out, CHART, LONG_VISIT)
    texts = " ".join(f["text"] for f in flags)
    assert "duration or onset" in texts
    assert "severity" in texts
