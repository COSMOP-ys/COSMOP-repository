"""Jev transport.

Jev is a decision-only model: you hand it state plus a typed question and it
returns one option out of a set you enumerated in advance, with a probability
attached. It generates no text, so anything free-form (a note body, a search
string) has to come from somewhere else.

Access, as of 2026-09-19: Jev is in beta on OpenRouter, model id
`~typesafe/jev-latest` (pinned: `typesafe/jev-1.13`), $0.042 / 1M input tokens
with output free, 32k context, 70-500 ms end to end.

NOTE ON THE WIRE FORMAT: openrouter.ai is unreachable from the sandbox this was
written in, so the request/response mapping below could not be exercised against
the live beta. Everything transport-specific is confined to `_REQUEST` /
`_parse_probabilities`; check those two against the current OpenRouter docs
before the first real call. The decision logic in `policy.py` and `pipeline.py`
does not depend on them.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Protocol, Sequence

from .schema import Decision

DEFAULT_MODEL = "~typesafe/jev-latest"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/decisions"


class Chooser(Protocol):
    """The slice of Jev this pipeline uses."""

    def choose(
        self, *, question: str, state: Mapping[str, Any], options: Sequence[str]
    ) -> Decision:
        """Pick one of `options` (one-of-N, N <= 255)."""

    def yes_no(self, *, question: str, state: Mapping[str, Any]) -> Decision:
        """Yes/no probability. `choice` is "yes" or "no"."""


class JevError(RuntimeError):
    def __init__(self, message: str, *, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


class OpenRouterJev:
    """Minimal Jev client over OpenRouter's decisions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 2.0,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise JevError("no API key: pass api_key= or set OPENROUTER_API_KEY")
        self._key = key
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout

    def choose(
        self, *, question: str, state: Mapping[str, Any], options: Sequence[str]
    ) -> Decision:
        if not 1 <= len(options) <= 255:
            raise JevError(f"one-of-N takes 1..255 options, got {len(options)}")
        if len(set(options)) != len(options):
            raise JevError("options must be unique")
        return self._post({"type": "choice", "options": list(options)}, question, state)

    def yes_no(self, *, question: str, state: Mapping[str, Any]) -> Decision:
        return self._post({"type": "boolean"}, question, state)

    def _post(
        self, answer: Mapping[str, Any], question: str, state: Mapping[str, Any]
    ) -> Decision:
        body = json.dumps(
            {"model": self._model, "question": question, "state": state, "answer": answer}
        ).encode()
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:  # surface the body, it explains the 4xx
            raise JevError(f"jev request failed: {exc.code}", payload=exc.read()) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JevError(f"jev request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        probabilities = _parse_probabilities(payload)
        choice = max(probabilities, key=probabilities.__getitem__)
        return Decision(choice=choice, probabilities=probabilities, latency_ms=latency_ms)


def _parse_probabilities(payload: Any) -> dict[str, float]:
    """Pull {option: probability} out of a decision response.

    Accepts a probability map, or a single choice plus a confidence (which is
    then spread over the remaining mass so callers see one shape either way).
    """
    if not isinstance(payload, Mapping):
        raise JevError("decision response was not an object", payload=payload)
    node = payload.get("decision", payload)
    if not isinstance(node, Mapping):
        raise JevError("decision field was not an object", payload=payload)

    for key in ("probabilities", "distribution"):
        raw = node.get(key)
        if isinstance(raw, Mapping) and raw:
            return {str(k): float(v) for k, v in raw.items()}

    choice = node.get("choice", node.get("value"))
    confidence = node.get("confidence", node.get("probability"))
    if choice is not None and confidence is not None:
        return {str(choice): float(confidence)}

    raise JevError("no probabilities in decision response", payload=payload)


class MockJev:
    """Offline stand-in. `handler` receives (question, state, options)."""

    def __init__(
        self, handler: Callable[[str, Mapping[str, Any], Sequence[str]], Decision]
    ) -> None:
        self._handler = handler
        self.calls: list[tuple[str, Sequence[str]]] = []

    def choose(
        self, *, question: str, state: Mapping[str, Any], options: Sequence[str]
    ) -> Decision:
        self.calls.append((question, list(options)))
        decision = self._handler(question, state, list(options))
        if decision.choice not in options:
            raise JevError(f"mock returned {decision.choice!r}, not one of {list(options)}")
        return decision

    def yes_no(self, *, question: str, state: Mapping[str, Any]) -> Decision:
        self.calls.append((question, ["yes", "no"]))
        return self._handler(question, state, ["yes", "no"])
