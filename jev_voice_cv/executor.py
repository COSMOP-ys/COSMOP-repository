"""Action execution, with the one piece of bookkeeping speculation needs.

A speculated read-only action usually runs again when the final transcript
confirms it. Re-running a read is harmless but wasteful (and visible, if it
scrolls or focuses something), so identical work inside a short window is
collapsed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .schema import Plan, PlanKind


class Executor(Protocol):
    def run(self, plan: Plan) -> str:
        """Perform the plan's action. Only called for runnable plans."""


def _key(plan: Plan) -> tuple[str | None, str | None]:
    return (plan.action, plan.target.ref if plan.target else None)


@dataclass
class Dedupe:
    """Collapses the speculate-then-confirm pair into one execution."""

    window_seconds: float = 3.0
    clock: Callable[[], float] = time.monotonic
    _seen: dict[tuple[str | None, str | None], float] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        """Forget history. Called between utterances, not between revisions."""
        self._seen.clear()

    def should_run(self, plan: Plan) -> bool:
        now = self.clock()
        key = _key(plan)
        last = self._seen.get(key)
        self._seen[key] = now
        return last is None or now - last > self.window_seconds


@dataclass
class DryRunExecutor:
    """Logs what would happen. The default until thresholds are tuned on real data."""

    dedupe: Dedupe = field(default_factory=Dedupe)
    log: list[str] = field(default_factory=list)

    def run(self, plan: Plan) -> str:
        if not plan.runnable:
            raise ValueError(f"not runnable: {plan.kind.value}")
        if not self.dedupe.should_run(plan):
            line = f"skip (already ran): {plan.action}"
            self.log.append(line)
            return line
        where = f" -> {plan.target.describe()}" if plan.target else ""
        arg = " [text arg from LLM]" if plan.text_arg_from else ""
        prefix = "speculate" if plan.kind is PlanKind.SPECULATE else "execute"
        line = f"{prefix}: {plan.action}{where}{arg} (p={plan.confidence:.2f})"
        self.log.append(line)
        return line
