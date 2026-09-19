"""voice -> (grounding) -> Jev -> gated action.

Speed in this design comes from deciding on partial transcripts, before the
speaker has finished. That is also where it can go wrong, so two rules are
structural rather than optional:

* every new transcript supersedes the previous one, and a result that comes back
  for a superseded ticket is discarded, never acted on;
* a partial transcript can only trigger a read-only, idempotent action, and only
  above a higher bar than the end-of-utterance path.

There are at most two round trips per revision. "Is this addressed to the
computer?" and "which action is it?" are different questions about the same
state, so they go in one request; Jev answers several questions in parallel and
bills input tokens only, which makes the filter question effectively free. The
target question is separate because the candidate set does not exist until an
action that needs one has been picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .grounding import Frame, Grounder
from .jev import Chooser, boolean, choice, score
from .policy import combine, gate
from .schema import ActionSpec, Candidate, Plan, PlanKind

NO_ACTION = "no_action"

_ADDRESSED = "Is the transcript an instruction addressed to this computer?"
_INTENT = "Which action does the speaker want?"
_TARGET = "Which on-screen candidate is the speaker referring to?"

# A second question about the same answer, asked because the first one refuses
# to vary: Jev's choice probability comes back at 1.00 for nearly everything,
# including answers that are wrong, so it cannot carry a threshold. `score` is
# the question type built to produce gradation, and measured against a labelled
# set it does - see the README. The rung wording is doing the work here; each
# one has to name a state a real utterance can be in, or the middle cases have
# nowhere to go and the scale collapses back to its endpoints.
CLARITY = (
    "How clearly does this utterance name one single action from the list, "
    "as an instruction to a computer?"
)
CLARITY_RUNGS: tuple[str, ...] = (
    "not an instruction at all: a question, an aside, or talk about something else",
    "an instruction, but which action is guesswork: it fits several equally, or none",
    "an unfinished instruction: the action is implied but the sentence stops short",
    "one action fits clearly, though the words are loose or indirect",
    "unmistakable: it names exactly one of these actions and nothing else",
)


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
    # Minimum position on the CLARITY_RUNGS scale before an action may run
    # unattended. None disables the question entirely; it rides along in the
    # request that is already being sent, so enabling it costs tokens and no
    # round trip. 2.5-3.0 is what the labelled set suggests, on 18 answers -
    # calibrate before trusting it.
    clarity_floor: float | None = None

    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("at least one action is required")
        self._by_name = {spec.name: spec for spec in self.actions}
        if len(self._by_name) != len(self.actions):
            raise ValueError("action names must be unique")
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
            return Plan(
                kind=PlanKind.SUPERSEDED, reason="newer transcript arrived", trace=tuple(trace)
            )

        def unmatched(reason: str, confidence: float, action: str | None = None) -> Plan:
            # Mid-utterance there is nothing to report yet, so wait for more audio;
            # at the end of one, say so rather than guessing.
            return Plan(
                kind=PlanKind.HOLD if not ticket.final else PlanKind.REJECT,
                action=action,
                confidence=confidence,
                reason=reason,
                trace=tuple(trace),
            )

        if self._stale(ticket):
            return superseded()

        options = {spec.name: spec.meaning for spec in self.actions}
        options[NO_ACTION] = "none of these; the speaker wants something else or nothing"
        questions = {"intent": choice(_INTENT, options)}
        if self.command_filter:
            questions["addressed"] = boolean(
                _ADDRESSED,
                true="a command meant for this machine",
                false="thinking aloud, or talk directed at another person",
            )
        if self.clarity_floor is not None:
            questions["clarity"] = score(CLARITY, CLARITY_RUNGS)

        answers = self.jev.evaluate(state=state, questions=questions)
        if self._stale(ticket):
            return superseded()

        if self.command_filter:
            addressed = answers["addressed"].probabilities.get("yes", 0.0)
            trace.append(f"addressed={addressed:.2f}")
            if addressed < self.command_threshold:
                return unmatched("not addressed to the computer", addressed)

        intent = answers["intent"]
        trace.append(f"intent={intent.choice}@{intent.confidence:.2f}")
        if intent.choice == NO_ACTION:
            return unmatched("no action matched", intent.confidence)

        clarity = answers["clarity"].score if "clarity" in answers else None
        if clarity is not None:
            trace.append(f"clarity={clarity:.2f}")

        spec = self._by_name[intent.choice]
        target: Candidate | None = None
        target_confidence: float | None = None

        if spec.needs_target:
            target, target_confidence, reason = self._resolve_target(spec, ticket, state, trace)
            if self._stale(ticket):
                return superseded()
            if target is None:
                # Never fall through to a guessed target: acting on the wrong
                # element is the failure mode voice control is judged on.
                return Plan(
                    kind=PlanKind.HOLD if not ticket.final else PlanKind.CONFIRM,
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
            clarity=clarity,
            clarity_floor=self.clarity_floor,
        )

    def _resolve_target(
        self,
        spec: ActionSpec,
        ticket: Ticket,
        state: Mapping[str, Any],
        trace: list[str],
    ) -> tuple[Candidate | None, float | None, str]:
        """Ground the utterance to one on-screen candidate, then let Jev pick."""
        if self.grounder is None:
            return None, None, "action needs a target but no grounder is configured"
        if getattr(self.grounder, "needs_frame", True) and ticket.frame is None:
            return None, None, "action needs a target but no frame was captured"

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
        pick = self.jev.evaluate(
            state=state,
            questions={"target": choice(_TARGET, {c.ref: c.describe() for c in candidates})},
        )["target"]
        trace.append(f"target={pick.choice}@{pick.confidence:.2f}")
        chosen = next((c for c in candidates if c.ref == pick.choice), None)
        if chosen is None:
            return None, None, f"model picked unknown candidate {pick.choice!r}"
        return chosen, pick.confidence, ""
