"""An example action set, sized the way the thresholds are meant to be sized.

Reads are cheap to get wrong and get a low bar plus permission to run on a
partial transcript. Writes cost a keystroke to undo and only run at the end of
the utterance. Anything irreversible always asks, however confident Jev is -
a type-valid decision can still be semantically wrong.
"""

from __future__ import annotations

from .schema import ActionSpec

BROWSER_ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec("scroll_to_element", read_only=True, idempotent=True,
               execute_threshold=0.70, speculate_threshold=0.88, needs_target=True),
    ActionSpec("highlight_element", read_only=True, idempotent=True,
               execute_threshold=0.70, speculate_threshold=0.88, needs_target=True),
    ActionSpec("open_app", read_only=True, idempotent=True,
               execute_threshold=0.80, speculate_threshold=0.93),
    ActionSpec("click_element", execute_threshold=0.90, speculate_threshold=0.99,
               needs_target=True),
    ActionSpec("type_text", execute_threshold=0.90, speculate_threshold=0.99,
               needs_text_arg=True),
    ActionSpec("create_note", execute_threshold=0.88, speculate_threshold=0.97,
               needs_text_arg=True),
    ActionSpec("delete_element", always_confirm=True, execute_threshold=0.95,
               speculate_threshold=0.99, needs_target=True),
    ActionSpec("submit_payment", always_confirm=True, execute_threshold=0.98,
               speculate_threshold=1.0),
)
