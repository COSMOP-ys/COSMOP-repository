"""CV grounding: spoken description -> concrete on-screen candidates.

This is the half of voice-operation accuracy that speech recognition cannot
fix. "click the red bicycle" fails not because the words were misheard but
because nothing resolved them to a region of the screen. SAM 3.1 takes a text
prompt plus an image or video frame and returns boxes and pixel masks for the
instances that match, tracking identity across frames (Object Multiplex, up to
16 targets at once). Jev then picks one of those candidates as a typed
one-of-N decision - a closed set, so there is no coordinate to hallucinate.

Cost, as of 2026-09-19: SAM 3.1 API is $2.50 / 1k images and $0.20 / 1k video
frames. Every call ships a screenshot off-device; see README before pointing
this at a screen with anything private on it.

Same caveat as jev.py: developer.meta.com was unreachable from the sandbox this
was written in, so `_REQUEST_PATH` / `_parse_candidates` are the two places to
check against the current API docs before the first real call.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .schema import Candidate

DEFAULT_MODEL = "sam-3.1"
DEFAULT_ENDPOINT = "https://api.meta.com/v1/segment"
MAX_MULTIPLEX_TARGETS = 16


@dataclass(frozen=True)
class Frame:
    """One screenshot or video frame."""

    image: bytes
    media_type: str = "image/png"
    width: int | None = None
    height: int | None = None


class Grounder(Protocol):
    # False for grounders that read the live page instead of a picture of it.
    # The pipeline only insists on a frame for the ones that need one.
    needs_frame: bool

    def ground(self, prompt: str, frame: Frame | None = None) -> Sequence[Candidate]:
        """Return the candidates matching `prompt`, best first."""


class GroundingError(RuntimeError):
    pass


class Sam31Grounder:
    """Text-prompted segmentation against the SAM 3.1 API.

    The fallback, not the default: see dom.py for why a browser should ground
    against the accessibility tree first and only reach for pixels when the
    target is a canvas, a video or a native window.
    """

    needs_frame = True

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        min_score: float = 0.5,
        max_candidates: int = MAX_MULTIPLEX_TARGETS,
        timeout: float = 5.0,
    ) -> None:
        key = api_key or os.environ.get("META_API_KEY")
        if not key:
            raise GroundingError("no API key: pass api_key= or set META_API_KEY")
        self._key = key
        self._model = model
        self._endpoint = endpoint
        self._min_score = min_score
        self._max_candidates = max_candidates
        self._timeout = timeout

    def ground(self, prompt: str, frame: Frame | None = None) -> Sequence[Candidate]:
        if frame is None:
            raise GroundingError("segmentation needs a frame")
        body = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "image": {
                    "media_type": frame.media_type,
                    "data": base64.b64encode(frame.image).decode(),
                },
                "max_objects": self._max_candidates,
            }
        ).encode()
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise GroundingError(f"sam request failed: {exc.code}: {exc.read()!r}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GroundingError(f"sam request failed: {exc}") from exc
        return self._rank(_parse_candidates(payload, prompt))

    def _rank(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Drop weak detections, best first, capped at the multiplex limit."""
        kept = [c for c in candidates if c.score is None or c.score >= self._min_score]
        kept.sort(key=lambda c: -(c.score or 0.0))
        return kept[: self._max_candidates]


def _parse_candidates(payload: Any, prompt: str) -> list[Candidate]:
    if not isinstance(payload, Mapping):
        raise GroundingError("segmentation response was not an object")
    raw: Iterable[Any] = payload.get("objects") or payload.get("instances") or ()
    candidates: list[Candidate] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        box = item.get("box") or item.get("bbox")
        candidates.append(
            Candidate(
                ref=str(item.get("id", f"obj-{index}")),
                label=str(item.get("label") or item.get("concept") or prompt),
                box=tuple(float(v) for v in box) if box and len(box) == 4 else None,
                score=float(item["score"]) if "score" in item else None,
            )
        )
    return candidates


class StaticGrounder:
    """Offline stand-in: returns fixtures, filtered by substring on the label.

    Kept for tests and fixtures; the real browser path is DomGrounder.
    """

    needs_frame = False

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self._candidates = list(candidates)
        self.prompts: list[str] = []

    def ground(self, prompt: str, frame: Frame | None = None) -> Sequence[Candidate]:
        self.prompts.append(prompt)
        words = [w for w in prompt.lower().split() if len(w) > 2]
        hits = [c for c in self._candidates if any(w in c.label.lower() for w in words)]
        return hits or []
