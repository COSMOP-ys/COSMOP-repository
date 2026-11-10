"""発音スコアの結果から、Claude で日本語学習者向けのコーチングを生成する。"""
from __future__ import annotations

import os

import anthropic

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM = (
    "あなたは日本語話者向けの英語発音コーチです。"
    "音素レベルの採点結果を受け取り、日本語で簡潔に助言してください。"
    "L/R、TH、V/B、母音の区別など日本人学習者に多い癖に結びつけ、"
    "口や舌の動かし方と、練習用の単語(ミニマルペア)を2〜3個示してください。"
    "採点は自動認識に基づくため誤認識の可能性がある点にも一言触れてください。"
    "励ましを忘れず、全体で300字程度に収めてください。"
)


def coach(text: str, score: float, mistakes: list[str]) -> str:
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY を環境変数から読む
    detail = "\n".join(f"- {m}" for m in mistakes) or "- 誤りは検出されませんでした"
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"練習した文: {text}\nスコア: {score}/100\n検出された誤り:\n{detail}",
            }
        ],
    )
    if resp.stop_reason == "refusal":
        return "(フィードバックを生成できませんでした。スコアと誤りの一覧を参考にしてください)"
    return "".join(b.text for b in resp.content if b.type == "text")
