"""Confidence gating: turns a Jev decision into "run it", "ask", or "wait".

This module is where voice-operation accuracy is actually won. Jev guarantees
the answer fits the schema, not that the answer is right - a type-valid
decision can still be semantically wrong. So every decision passes a gate that
is sized to the cost of that action being wrong.
"""

from __future__ import annotations

from .schema import ActionSpec, Candidate, Plan, PlanKind


def combine(intent_confidence: float, target_confidence: float | None) -> float:
    """Confidence of "right action AND right target".

    Multiplying treats the two decisions as independent, which they are not:
    Jev sees the same transcript both times, so correlated errors make this an
    optimistic estimate. Fine as a gate, not as a calibration claim - measure
    the joint rate on your own data before trusting the number itself.
    """
    if target_confidence is None:
        return intent_confidence
    return intent_confidence * target_confidence


def gate(
    spec: ActionSpec,
    confidence: float,
    *,
    final: bool,
    target: Candidate | None = None,
    trace: tuple[str, ...] = (),
) -> Plan:
    """Decide what to do with a scored action, at the given point in the utterance."""

    def plan(kind: PlanKind, reason: str) -> Plan:
        return Plan(
            kind=kind,
            action=spec.name,
            target=target,
            confidence=confidence,
            reason=reason,
            text_arg_from="llm" if spec.needs_text_arg else None,
            trace=trace,
        )

    if spec.needs_target and target is None:
        # Nothing on screen matched the spoken description. Guessing a target is
        # the worst failure mode of voice control, so never fall through to it.
        return plan(PlanKind.HOLD if not final else PlanKind.CONFIRM, "no grounded target")

    if confidence < spec.reject_floor:
        return plan(PlanKind.HOLD if not final else PlanKind.REJECT, "below reject floor")

    if not final:
        if not (spec.read_only and spec.idempotent):
            # Acting mid-sentence on anything that writes state is unrecoverable
            # when the sentence turns out to continue ("delete the file" ...
            # "-- no, keep it").
            return plan(PlanKind.HOLD, "partial transcript, action is not read-only+idempotent")
        if confidence < spec.speculate_threshold:
            return plan(PlanKind.HOLD, "partial transcript, below speculate threshold")
        return plan(PlanKind.SPECULATE, "read-only and confident on partial")

    if spec.always_confirm:
        return plan(PlanKind.CONFIRM, "action always requires confirmation")
    if confidence < spec.execute_threshold:
        return plan(PlanKind.CONFIRM, "below execute threshold")
    return plan(PlanKind.EXECUTE, "final transcript, above execute threshold")
