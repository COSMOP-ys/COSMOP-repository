"""First real call to Jev: does the wire format hold, how fast, how calibrated.

Three questions, in order of how much they matter:

1. Does the derived request actually work? The endpoint was read out of the AI
   SDK's gateway provider, not out of documentation, so the first 200 is the
   thing that turns it from "probably" into "yes".
2. How long does a decision take? The whole design spends round trips on
   partial transcripts, which only pays if a round trip is cheap.
3. Are the probabilities worth gating on? Every threshold in policy.py assumes
   a reported 0.90 means roughly nine times out of ten. TypeSafe publishes no
   calibration guarantee. This runs a small labelled set and prints the
   reliability table, so the thresholds stop being guesses.

    AI_GATEWAY_API_KEY=... python scripts/probe_jev.py
    AI_GATEWAY_API_KEY=... python scripts/probe_jev.py --raw --repeats 5

The set below is 24 utterances, which is enough to catch a badly-shaped
distribution and nowhere near enough to set a threshold on. Replace it with
your own recorded sessions before trusting a number.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import urllib.error
import urllib.request
from typing import Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jev_voice_cv.actions import BROWSER_ACTIONS  # noqa: E402
from jev_voice_cv.jev import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    GatewayJev,
    JevError,
    boolean,
    choice,
)
from jev_voice_cv.pipeline import NO_ACTION  # noqa: E402

ADDRESSED = "Is the transcript an instruction addressed to this computer?"
INTENT = "Which action does the speaker want?"

# (transcript, expected action or NO_ACTION, is it addressed to the computer)
CASES: Sequence[tuple[str, str, bool]] = (
    # Unambiguous commands.
    ("click the save changes button", "click_element", True),
    ("press submit", "click_element", True),
    ("scroll down to the pricing table", "scroll_to_element", True),
    ("show me the delete button", "scroll_to_element", True),
    ("highlight the email field", "highlight_element", True),
    ("where is the cancel button", "highlight_element", True),
    ("open the notes app", "open_app", True),
    ("switch to my calendar", "open_app", True),
    ("type my email address in there", "type_text", True),
    ("make a note about the meeting tomorrow", "create_note", True),
    ("delete this row", "delete_element", True),
    ("remove the attachment", "delete_element", True),
    ("pay for the order", "submit_payment", True),
    # Addressed, but not one of the actions.
    ("what time is it", NO_ACTION, True),
    ("how much does this cost", NO_ACTION, True),
    ("undo that", NO_ACTION, True),
    # Not addressed to the computer at all.
    ("so anyway I told him to open a ticket", NO_ACTION, False),
    ("she said we should delete the whole thing", NO_ACTION, False),
    ("by the way did you see the game last night", NO_ACTION, False),
    ("no no not that one", NO_ACTION, False),
    # Partial transcripts, the case speculation depends on.
    ("scroll to the", NO_ACTION, True),
    ("click the sa", "click_element", True),
    ("open the", "open_app", True),
    ("delete", "delete_element", True),
)


def _key_from_env_file() -> str | None:
    """Read the key out of a gitignored .env.local, so it never has to be typed
    into a shell that keeps history."""
    path = pathlib.Path(__file__).resolve().parent.parent / ".env.local"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "AI_GATEWAY_API_KEY":
            return value.strip().strip("\"'") or None
    return None


def questions():
    options = {spec.name: spec.meaning for spec in BROWSER_ACTIONS}
    options[NO_ACTION] = "none of these; the speaker wants something else or nothing"
    return {
        "intent": choice(INTENT, options),
        "addressed": boolean(
            ADDRESSED,
            true="a command meant for this machine",
            false="thinking aloud, or talk directed at another person",
        ),
    }


def raw_call(key: str, state, qs) -> tuple[int, dict]:
    """One unparsed call, so the response shape can be inspected as it arrives."""
    body = json.dumps(
        {"state": state, "questions": {k: q.to_wire() for k, q in qs.items()}}
    ).encode()
    request = urllib.request.Request(
        DEFAULT_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "ai-evaluation-model-specification-version": "4",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-model-id": DEFAULT_MODEL,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error_body": exc.read().decode("utf-8", "replace")}


def reliability(rows: list[tuple[float, bool]]) -> str:
    """Reported probability vs how often it was actually right."""
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    lines = ["  reported        n   said   actually right"]
    for low, high in buckets:
        inside = [ok for p, ok in rows if low <= p < high]
        if not inside:
            continue
        said = statistics.mean([p for p, _ in rows if low <= p < high])
        hit = sum(inside) / len(inside)
        flag = "" if abs(said - hit) < 0.15 else "   <-- off"
        lines.append(f"  {low:.2f}-{high:<5.2f}  {len(inside):3d}   {said:.2f}   {hit:.2f}{flag}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="store_true", help="dump the first raw response")
    parser.add_argument("--repeats", type=int, default=3, help="latency samples")
    args = parser.parse_args(argv)

    key = os.environ.get("AI_GATEWAY_API_KEY") or _key_from_env_file()
    if not key:
        print(
            "no key. Either export AI_GATEWAY_API_KEY, or write a line\n"
            "  AI_GATEWAY_API_KEY=...\n"
            "into .env.local at the repo root (gitignored).",
            file=sys.stderr,
        )
        return 2

    qs = questions()

    # 1. Wire format -------------------------------------------------------
    print(f"POST {DEFAULT_ENDPOINT}  ({DEFAULT_MODEL})")
    status, payload = raw_call(key, {"transcript": CASES[0][0], "transcript_is_final": True}, qs)
    print(f"  status {status}")
    if args.raw or status != 200:
        print(json.dumps(payload, indent=2)[:2000])
    if status != 200:
        print("\nthe derived request did not work; the body above says why", file=sys.stderr)
        return 1
    answers = payload.get("answers", {})
    print(f"  answers: {sorted(answers)}")
    print(f"  intent answer keys: {sorted(answers.get('intent', {}))}")
    print(f"  usage: {payload.get('usage')}   rounding: {payload.get('rounding')}")
    got_probs = isinstance(answers.get("intent", {}).get("probabilities"), dict)
    print(f"  choice returns a full distribution: {got_probs}"
          f"{'' if got_probs else '   <-- policy.py needs one; check the docs'}")

    # 2. Latency and 3. calibration ---------------------------------------
    jev = GatewayJev(key, timeout=20)
    latencies: list[float] = []
    intent_rows: list[tuple[float, bool]] = []
    addressed_rows: list[tuple[float, bool]] = []
    wrong: list[str] = []

    print(f"\n{len(CASES)} cases x {args.repeats} repeats")
    for transcript, expected, is_addressed in CASES:
        for attempt in range(args.repeats):
            try:
                result = jev.evaluate(
                    state={"transcript": transcript, "transcript_is_final": True}, questions=qs
                )
            except JevError as exc:
                print(f"  FAILED {transcript!r}: {exc}", file=sys.stderr)
                return 1
            latencies.append(result["intent"].latency_ms or 0.0)
            if attempt:
                continue
            intent, addressed = result["intent"], result["addressed"]
            ok = intent.choice == expected
            intent_rows.append((intent.confidence, ok))
            # Score the confidence in whichever way it leaned, against whether
            # leaning that way was right.
            p_yes = addressed.probabilities.get("yes", 0.0)
            addressed_rows.append((max(p_yes, 1 - p_yes), (p_yes >= 0.5) == is_addressed))
            if not ok:
                wrong.append(f"    {transcript!r}: said {intent.choice}@{intent.confidence:.2f}, "
                             f"expected {expected}")

    print(f"\nlatency  p50 {statistics.median(latencies):6.0f} ms   "
          f"min {min(latencies):.0f}   max {max(latencies):.0f}   n={len(latencies)}")

    hit = sum(ok for _, ok in intent_rows) / len(intent_rows)
    addressed_hit = sum(ok for _, ok in addressed_rows) / len(addressed_rows)
    print(f"\nintent    {hit:.0%} correct ({sum(ok for _, ok in intent_rows)}/{len(intent_rows)})")
    print(f"addressed {addressed_hit:.0%} correct")
    if wrong:
        print("  misses:")
        print("\n".join(wrong))

    print("\nintent calibration:")
    print(reliability(intent_rows))
    print("\naddressed calibration:")
    print(reliability(addressed_rows))
    print(
        "\n24 cases is a shape check, not a threshold. If a bucket is off, the gate in\n"
        "policy.py is reading the number as something it is not - retune against\n"
        "recorded sessions before trusting execute_threshold."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
