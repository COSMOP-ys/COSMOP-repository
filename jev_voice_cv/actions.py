"""An example action set, sized the way the thresholds are meant to be sized.

Reads are cheap to get wrong and get a low bar plus permission to run on a
partial transcript. Writes cost a keystroke to undo and only run at the end of
the utterance. Anything irreversible always asks, however confident Jev is -
a type-valid decision can still be semantically wrong.

The descriptions are not documentation. They are the option meanings Jev sees,
and they are the cheapest accuracy lever in the whole pipeline: rewriting one
costs nothing and changes which action wins on a borderline utterance.
"""

from __future__ import annotations

from .schema import ActionSpec

BROWSER_ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "scroll_to_element",
        "bring something already on the page into view, without interacting with it",
        read_only=True, idempotent=True,
        execute_threshold=0.70, speculate_threshold=0.88, needs_target=True,
    ),
    ActionSpec(
        "highlight_element",
        "point something out visually, so the speaker can confirm it is the right thing",
        read_only=True, idempotent=True,
        execute_threshold=0.70, speculate_threshold=0.88, needs_target=True,
    ),
    ActionSpec(
        "open_app",
        "launch or switch to an application or a website",
        read_only=True, idempotent=True,
        execute_threshold=0.80, speculate_threshold=0.93,
    ),
    ActionSpec(
        "click_element",
        "press a button, link or control on the screen",
        execute_threshold=0.90, speculate_threshold=0.99, needs_target=True,
    ),
    ActionSpec(
        "type_text",
        "enter dictated text into a named field, or into the focused one",
        execute_threshold=0.90, speculate_threshold=0.99,
        # People name the field they mean - "put my email in the email box" -
        # far more often than they rely on what happens to be focused. Saying
        # this action takes no target means the executor is handed nothing to
        # type into and fails at the last step of a decision that was right.
        needs_target=True, needs_text_arg=True,
    ),
    ActionSpec(
        "create_note",
        "write a new note or document containing dictated text",
        execute_threshold=0.88, speculate_threshold=0.97, needs_text_arg=True,
    ),
    ActionSpec(
        "delete_element",
        "remove or discard something; destructive and hard to undo",
        always_confirm=True,
        execute_threshold=0.95, speculate_threshold=0.99, needs_target=True,
    ),
    ActionSpec(
        "submit_payment",
        "confirm a purchase or send money",
        always_confirm=True,
        execute_threshold=0.98, speculate_threshold=1.0,
    ),
)
