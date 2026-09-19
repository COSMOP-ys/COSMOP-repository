"""Running a gated plan against a real page, through Playwright.

Everything upstream of here decides *what* to do. This is the part that does it,
and it only ever receives plans the gate already cleared, so it deliberately
contains no judgement of its own - no retries against a different element, no
"close enough" match, no falling back to coordinates.

Targets are addressed by the `data-jev-ref` attribute the snapshot stamped on
the element, not by re-running a description match. That matters: between
grounding and execution the page can move, and "the second red button" is a
different element than it was two hundred milliseconds ago. If the ref is gone,
the action fails loudly instead of hitting whatever took its place.

Playwright is imported lazily, so importing this module (and the package) works
without it installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .dom import DOM_SNAPSHOT_JS, DomNode
from .executor import Dedupe
from .schema import Plan

# Refs come back from the page, and a hostile page can put anything in an
# attribute. Anything that is not a plain handle never reaches a selector.
_SAFE_REF = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")

DEFAULT_TIMEOUT_MS = 2_000
Handler = Callable[[Any, Plan], str]


class ExecutionError(RuntimeError):
    pass


def selector_for(ref: str) -> str:
    if not _SAFE_REF.match(ref):
        raise ExecutionError(f"refusing to build a selector from {ref!r}")
    return f'[data-jev-ref="{ref}"]'


def snapshot(page: Any) -> list[DomNode]:
    """Current accessibility-ish snapshot of the page, as DomNodes."""
    return [DomNode.from_dict(raw) for raw in page.evaluate(DOM_SNAPSHOT_JS)]


def dom_provider(page: Any) -> Callable[[], list[DomNode]]:
    """A provider for `DomGrounder`, bound to this page."""
    return lambda: snapshot(page)


# -- default handlers --------------------------------------------------------


def _locator(page: Any, plan: Plan, timeout_ms: int):
    if plan.target is None:
        raise ExecutionError(f"{plan.action} needs a target and the plan carries none")
    locator = page.locator(selector_for(plan.target.ref))
    try:
        locator.wait_for(state="attached", timeout=timeout_ms)
    except Exception as exc:  # playwright raises its own TimeoutError
        raise ExecutionError(
            f"{plan.action}: {plan.target.ref} is no longer on the page"
        ) from exc
    return locator


def _text(plan: Plan) -> str:
    if not plan.text_arg:
        raise ExecutionError(
            f"{plan.action} needs text; Jev does not produce any, so call "
            "plan.with_text(...) before running it"
        )
    return plan.text_arg


_HIGHLIGHT_JS = """
(el, ms) => {
  const previous = el.style.outline;
  el.style.outline = '3px solid #ff3b30';
  el.style.outlineOffset = '2px';
  setTimeout(() => { el.style.outline = previous; }, ms);
}
"""


def scroll_to_element(page: Any, plan: Plan, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    _locator(page, plan, timeout_ms).scroll_into_view_if_needed(timeout=timeout_ms)
    return f"scrolled to {plan.target.ref}"


def highlight_element(
    page: Any, plan: Plan, *, timeout_ms: int = DEFAULT_TIMEOUT_MS, hold_ms: int = 1_500
) -> str:
    _locator(page, plan, timeout_ms).evaluate(_HIGHLIGHT_JS, hold_ms)
    return f"highlighted {plan.target.ref}"


def click_element(page: Any, plan: Plan, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    _locator(page, plan, timeout_ms).click(timeout=timeout_ms)
    return f"clicked {plan.target.ref}"


def type_text(page: Any, plan: Plan, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    text = _text(plan)
    if plan.target is not None:
        locator = _locator(page, plan, timeout_ms)
        try:
            locator.fill(text, timeout=timeout_ms)
        except Exception:
            # Not a fillable control (a contenteditable, a custom widget): focus
            # it and send keystrokes instead.
            locator.click(timeout=timeout_ms)
            page.keyboard.type(text)
        return f"typed {len(text)} chars into {plan.target.ref}"
    page.keyboard.type(text)
    return f"typed {len(text)} chars into the focused element"


DEFAULT_HANDLERS: Mapping[str, Handler] = {
    "scroll_to_element": scroll_to_element,
    "highlight_element": highlight_element,
    "click_element": click_element,
    "type_text": type_text,
}


# -- executor ----------------------------------------------------------------


@dataclass
class PlaywrightExecutor:
    """Executes plans against a Playwright `Page` (sync API).

    Only the actions in `handlers` can run. An action the pipeline can choose
    but this executor cannot perform fails as an unmistakable error rather than
    silently doing nothing - a voice interface that appears to ignore commands
    is worse than one that says it cannot.
    """

    page: Any
    handlers: dict[str, Handler] = field(default_factory=lambda: dict(DEFAULT_HANDLERS))
    dedupe: Dedupe = field(default_factory=Dedupe)
    log: list[str] = field(default_factory=list)

    def register(self, action: str, handler: Handler) -> None:
        """Teach the executor an app-specific action (open_app, save, ...)."""
        self.handlers[action] = handler

    def supports(self) -> Sequence[str]:
        return sorted(self.handlers)

    def run(self, plan: Plan) -> str:
        if not plan.runnable:
            raise ValueError(f"not runnable: {plan.kind.value}")
        handler = self.handlers.get(plan.action or "")
        if handler is None:
            raise ExecutionError(
                f"no handler for {plan.action!r}; registered: {list(self.supports())}"
            )
        if not self.dedupe.should_run(plan):
            line = f"skip (already ran): {plan.action}"
            self.log.append(line)
            return line
        line = handler(self.page, plan)
        self.log.append(line)
        return line
