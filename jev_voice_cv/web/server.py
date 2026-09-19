"""A local server that turns browser speech into gated plans.

The browser is the microphone and the screen; this process is the decision
layer. The page streams every revision of the transcript - Web Speech API
interim results, which arrive several times a second - together with a snapshot
of its own accessibility tree. Nothing but the transcript and the element names
ever leaves the machine, and with `--offline` not even that.

Run it:

    python -m jev_voice_cv.web.server              # needs AI_GATEWAY_API_KEY
    python -m jev_voice_cv.web.server --offline    # no key, no network

then open http://127.0.0.1:8765/.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

from ..actions import BROWSER_ACTIONS
from ..dom import DOM_SNAPSHOT_JS, DomGrounder, DomNode
from ..jev import GatewayJev, Question
from ..pipeline import NO_ACTION, VoicePipeline
from ..schema import Decision, Plan

HERE = pathlib.Path(__file__).parent
PAGE = HERE / "voice.html"
MAX_BODY = 2_000_000


class OfflineChooser:
    """Not a model. Keyword matching, so the page runs with no key and no network.

    It exists to exercise the loop - streaming partials, superseding, gating,
    executing - without a dependency on anything remote. Its confidences are
    made up, which is precisely why the real thing is worth measuring: swap in
    `GatewayJev` and the numbers stop being fiction.

    The cue lists carry Japanese as well as English. Jev itself needs none of
    this - it reads Japanese transcripts against these same English action
    descriptions - but a keyword stub is only as multilingual as its keywords.
    """

    _CUES: Mapping[str, tuple[str, ...]] = {
        "scroll_to_element": ("scroll", "show me", "go to", "find",
                              "スクロール", "まで移動"),
        "highlight_element": ("highlight", "point at", "point out", "where is",
                              "ハイライト", "どこ"),
        "open_app": ("open", "launch", "switch to",
                     "開い", "起動", "切り替"),
        "click_element": ("click", "press", "tap", "push", "hit",
                          "クリック", "押し", "押して", "タップ"),
        "type_text": ("type", "enter", "write", "put",
                      "入力", "打って", "書いて"),
        "create_note": ("note", "jot", "メモ"),
        "delete_element": ("delete", "remove", "discard",
                           "削除", "消し", "取り消"),
        "submit_payment": ("pay", "purchase", "buy", "支払", "購入"),
    }
    _NOT_A_COMMAND = (
        "i told", "he said", "she said", "they said", "anyway", "by the way",
        "って言っ", "と言っ", "そういえば", "ちなみに",
    )

    def evaluate(
        self, *, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> Mapping[str, Decision]:
        text = str(state.get("transcript", "")).lower()
        answers: dict[str, Decision] = {}
        for key, question in questions.items():
            if question.type == "boolean":
                p = 0.08 if any(c in text for c in self._NOT_A_COMMAND) else 0.93
                answers[key] = Decision("yes" if p >= 0.5 else "no", {"yes": p, "no": 1 - p})
            else:
                answers[key] = self._pick(text, list(question.criteria or []))
        return answers

    def _pick(self, text: str, options: Sequence[str]) -> Decision:
        hits = [o for o in options if any(cue in text for cue in self._CUES.get(o, ()))]
        if hits:
            winner = hits[0]
        elif NO_ACTION in options:
            winner = NO_ACTION
        else:
            # A candidate question: no cues apply, so trust the shortlist order.
            winner = options[0]
        # Split what is left evenly. The shape is a real distribution; the
        # numbers are not a measurement of anything.
        spare = 0.04 / max(len(options) - 1, 1)
        probabilities = {o: (0.96 if o == winner else spare) for o in options}
        return Decision(winner, probabilities, latency_ms=0.0)


def plan_to_json(plan: Plan) -> dict[str, Any]:
    return {
        "kind": plan.kind.value,
        "action": plan.action,
        "confidence": round(plan.confidence, 4),
        "reason": plan.reason,
        "runnable": plan.runnable,
        "needs_text": bool(plan.text_arg_from),
        "target": None
        if plan.target is None
        else {"ref": plan.target.ref, "label": plan.target.label, "score": plan.target.score},
        "trace": list(plan.trace),
    }


class Session:
    """One pipeline plus the snapshot the page last sent with a transcript."""

    def __init__(self, offline: bool) -> None:
        self._nodes: list[DomNode] = []
        jev = OfflineChooser() if offline else GatewayJev()
        self.pipeline = VoicePipeline(
            actions=BROWSER_ACTIONS,
            jev=jev,
            grounder=DomGrounder(lambda: self._nodes),
        )

    def resolve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._nodes = [DomNode.from_dict(n) for n in payload.get("nodes") or []]
        ticket = self.pipeline.submit(
            str(payload.get("text", "")), final=bool(payload.get("final"))
        )
        return plan_to_json(self.pipeline.resolve(ticket))

    def cancel(self) -> dict[str, Any]:
        self.pipeline.cancel()
        return {"kind": "superseded", "reason": "cancelled"}


class Handler(BaseHTTPRequestHandler):
    session: Session

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is served from here and talks only to here.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            # One source of truth for the extractor: the page gets the same JS
            # the Playwright path evaluates, injected at serve time.
            page = PAGE.read_text(encoding="utf-8").replace(
                "/*__DOM_SNAPSHOT_JS__*/null", DOM_SNAPSHOT_JS
            )
            self._send(200, page.encode(), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/cancel":
            self._json(200, self.session.cancel())
            return
        if self.path != "/resolve":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json(413, {"error": "snapshot too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        try:
            self._json(200, self.session.resolve(payload))
        except Exception as exc:  # a failed decision must not kill the server
            self._json(502, {"kind": "reject", "reason": f"{type(exc).__name__}: {exc}"})


def serve(host: str = "127.0.0.1", port: int = 8765, *, offline: bool = False) -> None:
    handler = type("BoundHandler", (Handler,), {"session": Session(offline)})
    server = ThreadingHTTPServer((host, port), handler)
    mode = "offline keyword stub" if offline else "Jev via AI Gateway"
    print(f"voice console on http://{host}:{port}/  ({mode})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the keyword stub instead of Jev (no API key, no network)",
    )
    args = parser.parse_args(argv)
    if not args.offline and not os.environ.get("AI_GATEWAY_API_KEY"):
        parser.error("set AI_GATEWAY_API_KEY, or pass --offline to run the stub")
    serve(args.host, args.port, offline=args.offline)


if __name__ == "__main__":
    main()
