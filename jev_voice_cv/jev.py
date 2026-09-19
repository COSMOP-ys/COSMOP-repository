"""Jev transport: typed questions in, probabilities out.

Jev is a decision-only model. You hand it shared state plus questions whose
answers are constrained to a shape you declared, and it returns choices, scores
and boolean probabilities. It generates no prose, so anything free-form (a note
body, a search string) has to come from somewhere else.

Access, as of 2026-09-19: Vercel AI Gateway, model id `typesafe-ai/jev`, 32k
context. List price is $0.042 / 1M input tokens with output free, and the model
is currently under promotional pricing at $0 in and out until 2026-09-25. It is
Free Tier eligible either way, so the $5/month included credit covers it - but
buying AI Gateway credits moves the team to the paid tier and the monthly free
credit stops applying, permanently.

WIRE FORMAT: Vercel documents evaluation as "available through the AI SDK only"
- there is no published REST endpoint for it. The request below was derived from
the AI SDK's own gateway provider rather than from docs, so it is pinned here
with the evidence:

    npm i @ai-sdk/gateway@4.0.87
    grep -n 'evaluation-model' node_modules/@ai-sdk/gateway/dist/index.js
    #   getUrl() { return `${this.config.baseURL}/evaluation-model`; }
    #   baseURL defaults to https://ai-gateway.vercel.sh/v4/ai
    #   body: { state, questions, ...providerOptions }
    #   headers: ai-evaluation-model-specification-version: 4, ai-model-id: <id>

and checked against the live service: a POST here with no credentials returns
401 `authentication_error`, while the same POST one character off the path
returns 404. Undocumented means it can move without a deprecation notice, so
`DEFAULT_ENDPOINT` and `_parse_answer` are the two places to re-check if calls
start failing. Nothing in policy.py or pipeline.py depends on either.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .schema import Decision

DEFAULT_MODEL = "typesafe-ai/jev"
DEFAULT_ENDPOINT = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
_SPECIFICATION_VERSION = "4"
_PROTOCOL_VERSION = "0.0.1"

# Jev's one-of-N ceiling. Past this, narrow the set before asking.
MAX_OPTIONS = 255


class JevError(RuntimeError):
    def __init__(self, message: str, *, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


@dataclass(frozen=True)
class Question:
    """One typed question. `criteria` says what the answer space means.

    boolean: {"true": ..., "false": ...}, or omitted.
    choice:  {option: what that option means}. The keys are the answer space.
    score:   an ordered sequence of rung labels, lowest first.
    """

    type: str
    instructions: str
    criteria: Mapping[str, str] | Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.type not in ("boolean", "choice", "score"):
            raise JevError(f"unknown question type {self.type!r}")
        if self.type == "choice":
            if not isinstance(self.criteria, Mapping) or not self.criteria:
                raise JevError("a choice question needs criteria mapping option -> meaning")
            if len(self.criteria) > MAX_OPTIONS:
                raise JevError(
                    f"one-of-N takes 1..{MAX_OPTIONS} options, got {len(self.criteria)}"
                )
        if self.type == "score":
            if not isinstance(self.criteria, Sequence) or isinstance(self.criteria, (str, bytes)):
                raise JevError("a score question needs an ordered sequence of rung labels")
            if len(self.criteria) < 2:
                raise JevError("a score question needs at least two rungs")

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            wire["criteria"] = (
                dict(self.criteria) if isinstance(self.criteria, Mapping) else list(self.criteria)
            )
        return wire


def boolean(instructions: str, *, true: str | None = None, false: str | None = None) -> Question:
    criteria = {k: v for k, v in (("true", true), ("false", false)) if v}
    return Question("boolean", instructions, criteria or None)


def choice(instructions: str, options: Mapping[str, str] | Iterable[str]) -> Question:
    """A closed answer set. Bare option names get themselves as their meaning.

    Prefer passing real descriptions: the model is picking between what the
    options mean, and an identifier like `scroll_to_element` says less than the
    sentence describing when to use it.
    """
    criteria = dict(options) if isinstance(options, Mapping) else {name: name for name in options}
    return Question("choice", instructions, criteria)


def score(instructions: str, rungs: Sequence[str]) -> Question:
    return Question("score", instructions, list(rungs))


class Chooser(Protocol):
    """The slice of Jev this pipeline uses."""

    def evaluate(
        self, *, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> Mapping[str, Decision]:
        """Answer every question against one state, in a single round trip."""

    def choose(
        self, *, question: str, state: Mapping[str, Any], options: Sequence[str]
    ) -> Decision: ...

    def yes_no(self, *, question: str, state: Mapping[str, Any]) -> Decision: ...


class _ChooserMixin:
    """`choose` / `yes_no` written in terms of `evaluate`, for single-question callers."""

    def evaluate(  # pragma: no cover - supplied by the concrete class
        self, *, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> Mapping[str, Decision]:
        raise NotImplementedError

    def choose(
        self, *, question: str, state: Mapping[str, Any], options: Sequence[str]
    ) -> Decision:
        if len(set(options)) != len(options):
            raise JevError("options must be unique")
        return self.evaluate(state=state, questions={"q": choice(question, options)})["q"]

    def yes_no(self, *, question: str, state: Mapping[str, Any]) -> Decision:
        return self.evaluate(state=state, questions={"q": boolean(question)})["q"]


class GatewayJev(_ChooserMixin):
    """Jev over Vercel AI Gateway's evaluation endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 2.0,
    ) -> None:
        key = api_key or os.environ.get("AI_GATEWAY_API_KEY")
        if not key:
            raise JevError("no API key: pass api_key= or set AI_GATEWAY_API_KEY")
        self._key = key
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout

    def evaluate(
        self, *, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> Mapping[str, Decision]:
        if not questions:
            raise JevError("no questions to evaluate")
        body = json.dumps(
            {"state": dict(state), "questions": {k: q.to_wire() for k, q in questions.items()}}
        ).encode()
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "ai-evaluation-model-specification-version": _SPECIFICATION_VERSION,
                "ai-gateway-protocol-version": _PROTOCOL_VERSION,
                "ai-model-id": self._model,
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
        return _parse_response(payload, questions, latency_ms)


def _parse_response(
    payload: Any, questions: Mapping[str, Question], latency_ms: float
) -> dict[str, Decision]:
    if not isinstance(payload, Mapping):
        raise JevError("evaluation response was not an object", payload=payload)
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise JevError("evaluation response had no answers object", payload=payload)
    missing = sorted(set(questions) - set(answers))
    if missing:
        raise JevError(f"no answer for {missing}", payload=payload)
    return {key: _parse_answer(answers[key], latency_ms) for key in questions}


def _parse_answer(answer: Any, latency_ms: float) -> Decision:
    """One answer -> Decision. Booleans become a yes/no distribution."""
    if not isinstance(answer, Mapping):
        raise JevError("answer was not an object", payload=answer)
    kind = answer.get("type")

    if kind == "boolean":
        p = float(answer["probability"])
        return Decision(
            choice="yes" if p >= 0.5 else "no",
            probabilities={"yes": p, "no": 1.0 - p},
            latency_ms=latency_ms,
        )

    if kind in ("choice", "score"):
        raw = answer.get("probabilities") or {}
        probabilities = {str(k): float(v) for k, v in raw.items()}
        if kind == "choice":
            picked = str(answer["choice"])
        else:
            # A score has no named winner; the top rung stands in as the choice
            # so callers that only read `confidence` still see the mass on it.
            if not probabilities:
                raise JevError("score answer carried no probabilities", payload=answer)
            picked = max(probabilities, key=probabilities.__getitem__)
        # Rounding can drop a near-zero entry; keep the shape consistent.
        probabilities.setdefault(picked, 0.0)
        return Decision(
            choice=picked,
            probabilities=probabilities,
            score=float(answer["score"]) if kind == "score" else None,
            latency_ms=latency_ms,
        )

    raise JevError(f"unknown answer type {kind!r}", payload=answer)


class MockJev(_ChooserMixin):
    """Offline stand-in. `handler` receives (instructions, state, options)."""

    def __init__(
        self, handler: Callable[[str, Mapping[str, Any], Sequence[str]], Decision]
    ) -> None:
        self._handler = handler
        self.calls: list[tuple[str, Sequence[str]]] = []
        self.batches: list[list[str]] = []

    def evaluate(
        self, *, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> Mapping[str, Decision]:
        self.batches.append(list(questions))
        answers: dict[str, Decision] = {}
        for key, question in questions.items():
            options = (
                ["yes", "no"] if question.type == "boolean" else list(question.criteria or [])
            )
            self.calls.append((question.instructions, options))
            decision = self._handler(question.instructions, state, options)
            if decision.choice not in options:
                raise JevError(f"mock returned {decision.choice!r}, not one of {options}")
            answers[key] = decision
        return answers
