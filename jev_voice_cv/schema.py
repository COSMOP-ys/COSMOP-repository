"""Types shared by the voice -> CV grounding -> Jev decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ActionSpec:
    """One entry of the closed action set Jev is allowed to choose from.

    Thresholds are per action on purpose: a single global cutoff either blocks
    harmless reads or lets a destructive action through on a mishearing.
    Scale them to what being wrong costs for that action.
    """

    name: str
    read_only: bool = False
    idempotent: bool = False
    # End-of-utterance bar. Below it we ask instead of acting.
    execute_threshold: float = 0.85
    # Bar for acting on a partial transcript, before the speaker has finished.
    # Strictly higher than execute_threshold: the sentence can still change.
    speculate_threshold: float = 0.95
    # Under this, do not even offer the action - treat as "not understood".
    reject_floor: float = 0.40
    # Destructive / irreversible: never auto-execute, whatever the confidence.
    always_confirm: bool = False
    # Needs a concrete on-screen target, resolved by CV grounding.
    needs_target: bool = False
    # Needs free-form text (note body, search query). Jev cannot produce text;
    # this part is routed to a small LLM or taken from the raw transcript.
    needs_text_arg: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.reject_floor <= self.execute_threshold <= self.speculate_threshold <= 1.0:
            raise ValueError(
                f"{self.name}: expected reject_floor <= execute_threshold <= "
                "speculate_threshold within [0, 1]"
            )
        if self.speculate_threshold == self.execute_threshold and self.read_only:
            raise ValueError(f"{self.name}: speculating on partials needs a higher bar than end-of-turn")


@dataclass(frozen=True)
class Candidate:
    """A grounded on-screen target: one SAM 3.1 instance, or one DOM node."""

    ref: str
    label: str
    box: tuple[float, float, float, float] | None = None
    score: float | None = None

    def describe(self) -> str:
        where = "" if self.box is None else f" at {tuple(round(v, 3) for v in self.box)}"
        return f"{self.label}{where}"


@dataclass(frozen=True)
class Decision:
    """A Jev typed decision: one option out of a known set, plus probabilities."""

    choice: str
    probabilities: Mapping[str, float]
    latency_ms: float | None = None

    @property
    def confidence(self) -> float:
        return self.probabilities.get(self.choice, 0.0)


class PlanKind(str, Enum):
    EXECUTE = "execute"        # final transcript, confident enough: run it
    SPECULATE = "speculate"    # partial transcript, read-only + very confident
    CONFIRM = "confirm"        # understood, but the user has to approve
    REJECT = "reject"          # below the floor: ask them to repeat
    HOLD = "hold"              # partial, not speculatable yet: wait for more audio
    SUPERSEDED = "superseded"  # a newer transcript arrived; this result is stale


@dataclass(frozen=True)
class Plan:
    kind: PlanKind
    action: str | None = None
    target: Candidate | None = None
    confidence: float = 0.0
    reason: str = ""
    # Set when the action needs free text that Jev cannot produce.
    text_arg_from: str | None = None
    trace: Sequence[str] = field(default_factory=tuple)

    @property
    def runnable(self) -> bool:
        return self.kind in (PlanKind.EXECUTE, PlanKind.SPECULATE)
