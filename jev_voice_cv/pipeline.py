"""voice -> (CV grounding) -> Jev -> gated action.

Speed in this design comes from deciding on partial transcripts, before the
speaker has finished. That is also where it can go wrong, so two rules are
structural rather than optional:

* every new transcript supersedes the previous one, and a result that comes back
  for a superseded ticket is discarded, never acted on;
* a partial transcript can only trigger a read-only, idempotent action, and only
  above a higher bar than the end-of-utterance path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .grounding import Frame, Grounder
from .jev import Chooser
from .policy import combine, gate
from .schema import ActionSpec, Candidate, Plan, PlanKind

NO_ACTION = "no_action"


@dataclass(frozen=True)
class Ticket:
    """One transcript revision handed to the pipeline."""

    seq: int
    text: str
    final: bool
    frame: Frame | None = None


@dataclass
class VoicePipeline:
    actions: Sequence[ActionSpec]
    jev: Chooser
    grounder: Grounder | None = None
    # App/screen state Jev gets alongside the question (open app, URL, selection).
    state_provider: Callable[[], Mapping[str, Any]] = dict
    # Reject utterances not addressed to the computer at all. Room noise and side
    # conversation cause more wrong actions than misheard commands do.
    command_filter: bool = True
    command_threshold: float = 0.60

    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("at least one action is required")
        self._by_name = {spec.name: spec for spec in self.actions}
        if NO_ACTION in self._by_name:
            raise ValueError(f"{NO_ACTION!r} is reserved as the reject option")

    # -- transcript bookkeeping ------------------------------------------------

    def submit(self, text: str, *, final: bool, frame: Frame | None = None) -> Ticket:
        """Register a transcript revision. Supersedes anything already in flight."""
        self._seq += 1
        return Ticket(seq=self._seq, text=text, final=final, frame=frame)

    def cancel(self) -> None:
        """Drop everything in flight ("no, stop", a released push-to-talk key)."""
        self._seq += 1

    def _stale(self, ticket: Ticket) -> bool:
        return ticket.seq != self._seq

    # -- decision -------------------------------------------------------------

    def resolve(self, ticket: Ticket) -> Plan:
        """Score a ticket and return what to do about it.

        Safe to call late: if newer audio has arrived since `submit`, the result
        is reported as superseded instead of being acted on.
        """
        trace: list[str] = []
        state = dict(self.state_provider())
        state["transcript"] = ticket.text
        state["transcript_is_final"] = ticket.final

        def superseded() -> Plan:
            return Plan(kind=PlanKind.SUPERSEDED, reason="newer transcript arrived", trace=tuple(trace))

        if self._stale(ticket):
            return superseded()

        if self.command_filter:
            verdict = self.jev.yes_no(
                question="Is the transcript an instruction addressed to this computer?",
                state=state,
            )
            if self._stale(ticket):
                return superseded()
            addressed = verdict.probabilities.get("yes", 0.0)
            trace.append(f"addressed={addressed:.2f}")
            if addressed < self.command_threshold:
                kind = PlanKind.HOLD if not ticket.final else PlanKind.REJECT
                return Plan(kind=kind, confidence=addressed, reason="not addressed to the computer", trace=tuple(trace))

        options = [spec.name for spec in self.actions] + [NO_ACTION]
        intent = self.jev.choose(
            question="Which action does the speaker want?", state=state, options=options
        )
        if self._stale(ticket):
            return superseded()
        trace.append(f"intent={intent.choice}@{intent.confidence:.2f}")

        if intent.choice == NO_ACTION:
            kind = PlanKind.HOLD if not ticket.final else PlanKind.REJECT
            return Plan(kind=kind, confidence=intent.confidence, reason="no action matched", trace=tuple(trace))

        spec = self._by_name[intent.choice]
        target: Candidate | None = None
        target_confidence: float | None = None

        if spec.needs_target:
            target, target_confidence, reason = self._resolve_target(spec, ticket, state, trace)
            if self._stale(ticket):
                return superseded()
            if target is None:
                kind = PlanKind.HOLD if not ticket.final else PlanKind.CONFIRM
                return Plan(
                    kind=kind,
                    action=spec.name,
                    confidence=intent.confidence,
                    reason=reason,
                    trace=tuple(trace),
                )

        return gate(
            spec,
            combine(intent.confidence, target_confidence),
            final=ticket.final,
            target=target,
            trace=tuple(trace),
        )

    def _resolve_target(
        self,
        spec: ActionSpec,
        ticket: Ticket,
        state: dict[str, Any],
        trace: list[str],
    ) -> tuple[Candidate | None, float | None, str]:
        """Ground the utterance to one on-screen candidate via CV, then Jev."""
        if self.grounder is None or ticket.frame is None:
            return None, None, "action needs a target but no grounder/frame available"

        candidates = list(self.grounder.ground(ticket.text, ticket.frame))
        trace.append(f"candidates={len(candidates)}")
        if not candidates:
            return None, None, "no grounded target"
        if len(candidates) == 1:
            only = candidates[0]
            # A single match still carries the detector's own uncertainty.
            return only, only.score if only.score is not None else 1.0, ""

        if self._stale(ticket):
            return None, None, "superseded"

        state = dict(state)
        state["candidates"] = [
            {"ref": c.ref, "label": c.label, "box": c.box, "detector_score": c.score}
            for c in candidates
        ]
        pick = self.jev.choose(
            question="Which candidate is the speaker pointing at?",
            state=state,
            options=[c.ref for c in candidates],
        )
        trace.append(f"target={pick.choice}@{pick.confidence:.2f}")
        chosen = next(c for c in candidates if c.ref == pick.choice)
        return chosen, pick.confidence, ""
