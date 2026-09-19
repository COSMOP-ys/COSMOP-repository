"""Demo runner: streams partial transcripts through the pipeline and prints the gate.

    python3 -m jev_voice_cv.cli            # offline, deterministic fake model
    python3 -m jev_voice_cv.cli --live     # real Jev + SAM 3.1 (needs API keys)

The offline model is a keyword fake, not a stand-in for Jev's accuracy. What the
demo shows is the control flow: which revisions act, which wait, which ask.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping, Sequence

from .actions import BROWSER_ACTIONS
from .executor import DryRunExecutor
from .grounding import Frame, StaticGrounder
from .jev import MockJev
from .pipeline import NO_ACTION, VoicePipeline
from .schema import Candidate, Decision, PlanKind

SCREEN = (
    Candidate("btn-1", "red delete button", (0.8, 0.1, 0.9, 0.15), score=0.94),
    Candidate("btn-2", "red submit button", (0.8, 0.3, 0.9, 0.35), score=0.91),
    Candidate("row-7", "shopping list row", (0.1, 0.5, 0.6, 0.55), score=0.88),
)

# Markers of speech about the computer rather than to it. The real pipeline asks
# Jev for a calibrated yes/no instead of matching strings.
ASIDE_MARKERS = ("i told", "he said", "she said", "they said", "anyway")

KEYWORDS = {
    "open_app": ("open", "launch"),
    "create_note": ("note", "list", "write down"),
    "scroll_to_element": ("scroll", "show me", "find"),
    "click_element": ("click", "press", "tap"),
    "delete_element": ("delete", "remove"),
}

UTTERANCES: tuple[tuple[str, ...], ...] = (
    ("open", "open the notes", "open the notes app", "open the notes app and create a shopping list"),
    ("scroll", "scroll to the red", "scroll to the red submit button"),
    ("delete", "delete the red", "delete the red delete button"),
    ("so anyway", "so anyway I told him", "so anyway I told him to open a ticket"),
)


def fake_jev(question: str, state: Mapping[str, Any], options: Sequence[str]) -> Decision:
    """Deterministic stand-in: keyword match, confidence rising with transcript length."""
    text = str(state.get("transcript", "")).lower()
    words = len(text.split())
    certainty = min(0.55 + 0.12 * words, 0.97)

    if options == ["yes", "no"]:
        commanding = any(k in text for ks in KEYWORDS.values() for k in ks)
        aside = any(m in text for m in ASIDE_MARKERS)
        addressed = 0.95 if commanding and not aside else 0.12
        return Decision("yes" if addressed >= 0.5 else "no", {"yes": addressed, "no": 1 - addressed})

    if "candidate" in question.lower():
        hits = [ref for ref in options if _overlap(_label(state, ref), text)]
        pick = hits[0] if hits else options[0]
        spread = certainty if hits else 0.5
        rest = (1 - spread) / max(len(options) - 1, 1)
        return Decision(pick, {ref: spread if ref == pick else rest for ref in options})

    scored = {
        name: certainty if any(k in text for k in keys) else 0.02
        for name, keys in KEYWORDS.items()
    }
    best_action = max(scored, key=scored.__getitem__)
    if scored[best_action] < 0.1:
        return Decision(NO_ACTION, {opt: 0.9 if opt == NO_ACTION else 0.01 for opt in options})
    rest = (1 - scored[best_action]) / max(len(options) - 1, 1)
    return Decision(
        best_action,
        {opt: scored[best_action] if opt == best_action else rest for opt in options},
    )


def _label(state: Mapping[str, Any], ref: str) -> str:
    for candidate in state.get("candidates", ()):
        if candidate["ref"] == ref:
            return str(candidate["label"]).lower()
    return ""


def _overlap(label: str, text: str) -> bool:
    words = [w for w in label.split() if len(w) > 2]
    return bool(words) and all(w in text for w in words)


def run_offline() -> int:
    pipeline = VoicePipeline(
        actions=BROWSER_ACTIONS,
        jev=MockJev(fake_jev),
        grounder=StaticGrounder(SCREEN),
        state_provider=lambda: {"focused_app": "Safari", "url": "https://example.com/cart"},
    )
    executor = DryRunExecutor()
    frame = Frame(image=b"", media_type="image/png", width=1512, height=982)

    for utterance in UTTERANCES:
        print(f"\n--- {utterance[-1]!r}")
        executor.dedupe.reset()
        for index, revision in enumerate(utterance):
            final = index == len(utterance) - 1
            ticket = pipeline.submit(revision, final=final, frame=frame)
            plan = pipeline.resolve(ticket)
            mark = "final" if final else "partial"
            print(f"  [{mark:7}] {revision!r:52} -> {plan.kind.value:10} {plan.reason}")
            if plan.runnable:
                print(f"            {executor.run(plan)}")
            elif plan.kind is PlanKind.CONFIRM:
                print(f"            awaiting confirmation: {plan.action} (p={plan.confidence:.2f})")
    return 0


def run_live() -> int:
    from .grounding import Sam31Grounder
    from .jev import OpenRouterJev

    pipeline = VoicePipeline(
        actions=BROWSER_ACTIONS, jev=OpenRouterJev(), grounder=Sam31Grounder()
    )
    text = input("transcript> ")
    plan = pipeline.resolve(pipeline.submit(text, final=True))
    print(plan)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call the real APIs (needs keys)")
    args = parser.parse_args(argv)
    return run_live() if args.live else run_offline()


if __name__ == "__main__":
    sys.exit(main())
