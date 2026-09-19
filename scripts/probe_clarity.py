"""Can a `score` question supply the gradation `choice` refuses to?

The measured problem: Jev's choice probability is saturated. 18 of 24 answers
came back at 1.00 and one in nine of those was wrong, so no `execute_threshold`
between 0.70 and 0.99 separates anything, and the two answers that should never
have run both reported 1.00.

The idea here is to stop asking one question to do two jobs. `choice` is decent
at *which* action (83-88%); it is the *how sure* that is useless. `score` is
the question type built to produce gradation - an interpolated position on an
ordered scale - so ask it separately:

    choice:  which of these actions does the speaker want?
    score:   how clearly does this utterance name one single action?

and gate on the second. This measures whether that actually separates the
right answers from the wrong ones, which is the only thing that would justify
the extra question.

    python scripts/probe_clarity.py              # one request, fits the free tier
    python scripts/probe_clarity.py --per-utterance
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from typing import Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from jev_voice_cv.actions import BROWSER_ACTIONS  # noqa: E402
from jev_voice_cv.jev import GatewayJev, JevError, boolean, choice, score  # noqa: E402
from jev_voice_cv.pipeline import (  # noqa: E402
    CLARITY as PIPELINE_CLARITY,
    CLARITY_RUNGS as PIPELINE_RUNGS,
    NO_ACTION,
)
from probe_jev import CASES, INTENT, _key_from_env_file  # noqa: E402

# One source of truth: the pipeline asks this exact question when
# clarity_floor is set, so the probe must not drift from it.
CLARITY, RUNGS = PIPELINE_CLARITY, PIPELINE_RUNGS



def options() -> dict[str, str]:
    opts = {spec.name: spec.meaning for spec in BROWSER_ACTIONS}
    opts[NO_ACTION] = "none of these; the speaker wants something else or nothing"
    return opts


def ask_batched(jev: GatewayJev):
    state = {"utterances": {f"u{i}": text for i, (text, _, _) in enumerate(CASES)}}
    qs = {}
    for i in range(len(CASES)):
        qs[f"intent_{i}"] = choice(f"Considering only utterance u{i}: {INTENT}", options())
        qs[f"clarity_{i}"] = score(f"Considering only utterance u{i}: {CLARITY}", RUNGS)
    answers = jev.evaluate(state=state, questions=qs)
    return [
        (answers[f"intent_{i}"], answers[f"clarity_{i}"]) for i in range(len(CASES))
    ]


def ask_per_utterance(jev: GatewayJev, delay: float):
    out = []
    for text, _, _ in CASES:
        answers = jev.evaluate(
            state={"transcript": text, "transcript_is_final": True},
            questions={
                "intent": choice(INTENT, options()),
                "clarity": score(CLARITY, RUNGS),
                # Kept so the request matches what the pipeline would send.
                "addressed": boolean(
                    "Is the transcript an instruction addressed to this computer?",
                    true="a command meant for this machine",
                    false="thinking aloud, or talk directed at another person",
                ),
            },
        )
        out.append((answers["intent"], answers["clarity"]))
        time.sleep(delay)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-utterance", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args(argv)

    key = _key_from_env_file()
    if not key:
        print("no GITHUB-style key: set AI_GATEWAY_API_KEY in .env.local", file=sys.stderr)
        return 2
    jev = GatewayJev(key, timeout=30, max_retries=10)

    try:
        pairs = (
            ask_per_utterance(jev, args.delay) if args.per_utterance else ask_batched(jev)
        )
    except JevError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    rows = []
    for (text, expected, _), (intent, clarity) in zip(CASES, pairs):
        rows.append((text, expected, intent.choice, intent.confidence, clarity.score or 0.0))

    top = len(RUNGS) - 1
    print(f"\nintent {sum(1 for r in rows if r[2] == r[1])}/{len(rows)} correct;"
          f"  clarity scale 0-{top}\n")
    print(f"{'utterance':<46}{'said':<20}{'p':>6}{'clarity':>9}")
    for text, expected, said, p, sc in rows:
        mark = "  " if said == expected else " x"
        print(f"{mark}{text[:44]:<44}{said:<20}{p:>6.2f}{sc:>9.2f}")

    # Only answers that picked an action can be judged here. Blocking a
    # `no_action` answer costs nothing - the pipeline already rejects it - so
    # counting those as "correct answers a bar would have blocked" makes any
    # bar look far worse than it is. That mistake is easy to make and it
    # reverses the conclusion.
    actionable = [r for r in rows if r[2] != NO_ACTION]
    right = [r for r in actionable if r[2] == r[1]]
    wrong = [r for r in actionable if r[2] != r[1]]
    if not wrong:
        print("\nno misses among the answers that would have run; nothing to separate")
        return 0

    print(f"\nof the {len(actionable)} answers that would have run something: "
          f"{len(right)} correct, {len(wrong)} wrong")
    print(f"  clarity   correct mean {statistics.mean(s for *_, s in right):.2f}"
          f"   wrong mean {statistics.mean(s for *_, s in wrong):.2f}")
    print(f"  choice p  correct mean {statistics.mean(p for *_, p, _ in right):.2f}"
          f"   wrong mean {statistics.mean(p for *_, p, _ in wrong):.2f}")

    print("\nwhat a clarity bar would cost and buy:")
    print("  bar    wrong blocked   correct blocked")
    for bar in [x / 2 for x in range(1, 2 * top + 1)]:
        blocked_wrong = sum(1 for *_, s in wrong if s < bar)
        blocked_right = sum(1 for *_, s in right if s < bar)
        print(f"  {bar:<6.1f} {blocked_wrong:>7}/{len(wrong):<8} {blocked_right:>7}/{len(right)}")

    print("\nranked by clarity - the order is the whole question:")
    for text, expected, said, p, sc in sorted(actionable, key=lambda r: r[4]):
        print(f"  {sc:5.2f}  p={p:.2f}  {'WRONG' if said != expected else 'ok   '}  {text}")
    print(
        "\nA bar is worth having only where it blocks wrong answers faster than\n"
        "correct ones. If the two columns climb together, the score is no better\n"
        "than the saturated probability it was meant to replace."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
