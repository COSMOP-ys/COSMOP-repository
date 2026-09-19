"""Getting the dictated words out of an utterance, without generating any.

Jev returns choices and probabilities, never prose, so "type my email in the
email box" leaves the pipeline knowing the action and the field and not the
text. The usual answer is to route that part to a small language model, which
means a second provider, a second latency budget and a second thing that can
hallucinate the contents of what you are about to type.

There is a cheaper one. The dictated text is almost always a contiguous run of
words the speaker already said, so enumerate those runs locally and let Jev
pick one. That is a one-of-N over a closed set - exactly the shape it answers -
and the answer is a substring of the transcript by construction. It cannot
invent an address, a name or an amount, because nothing here can produce a
character the speaker did not say.

What it cannot do: text that is not contiguous in the transcript, or that needs
rewriting (spelling out an address, expanding "at" to "@"). `NO_TEXT` is the
escape hatch for the first; the second still needs a model that writes.
"""

from __future__ import annotations

import re
from typing import Sequence

# Runs of writing, with everything between them treated as a separator. Same
# script-boundary idea as the grounder, so Japanese splits without spaces.
_PIECE = re.compile(
    r"[一-鿿㐀-䶿]+"
    r"|[゠-ヿｦ-ﾟー]+"
    r"|[぀-ゟ]+"
    r"|[A-Za-z0-9][A-Za-z0-9'@._+-]*"
)

# The option meaning "the words to enter are not in what was said".
NO_TEXT = "__no_text__"

MAX_PIECES = 24
MAX_SPAN_PIECES = 12
MAX_OPTIONS = 200


def spans(
    transcript: str,
    *,
    max_pieces: int = MAX_PIECES,
    max_span_pieces: int = MAX_SPAN_PIECES,
    max_options: int = MAX_OPTIONS,
) -> list[str]:
    """Every contiguous run of words in `transcript`, longest first.

    Longest first because the cap bites at the end, and a long span is the one
    a short one can still be recovered from by a person reading the result.
    """
    pieces = [m.span() for m in _PIECE.finditer(transcript)][:max_pieces]
    if not pieces:
        return []

    seen: dict[str, None] = {}
    out: list[tuple[int, str]] = []
    for length in range(min(max_span_pieces, len(pieces)), 0, -1):
        for start in range(0, len(pieces) - length + 1):
            text = transcript[pieces[start][0]:pieces[start + length - 1][1]].strip()
            if text and text not in seen:
                seen[text] = None
                out.append((length, text))
    return [text for _, text in out][:max_options]


def options(transcript: str, **kwargs) -> dict[str, str]:
    """Spans as a Jev choice set, plus the escape hatch.

    Each span is its own description: there is nothing to say about "buy milk"
    beyond the words themselves.
    """
    chosen = {text: text for text in spans(transcript, **kwargs)}
    chosen[NO_TEXT] = "the words to enter were not actually said in this utterance"
    return chosen


def question_for(action: str) -> str:
    return (
        f"The speaker wants to {action}. Which part of the transcript is the text "
        "to enter - the content itself, not the words that asked for it?"
    )


def resolve(choice_value: str, candidates: Sequence[str]) -> str | None:
    """The chosen span, or None when the model took the escape hatch."""
    if choice_value == NO_TEXT or choice_value not in candidates:
        return None
    return choice_value
