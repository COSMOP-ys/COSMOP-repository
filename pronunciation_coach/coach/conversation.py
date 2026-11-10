"""会話練習: 音声 → 文字起こし → Claude が英語で返答 + 日本語で添削。"""
from __future__ import annotations

import os

import anthropic

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
WHISPER_ID = os.environ.get("WHISPER_MODEL", "openai/whisper-small.en")
SEP = "===FEEDBACK==="
_cache: dict = {}

LEVELS = {
    "初級": "Use very simple words and short sentences (CEFR A2).",
    "中級": "Use everyday vocabulary and natural sentences (CEFR B1-B2).",
    "上級": "Use natural, idiomatic, richer language (CEFR C1).",
}

SYSTEM_TMPL = (
    "You are a friendly English conversation partner for a Japanese learner. "
    "Topic/scenario: {topic}. {level} "
    "Stay in character, keep replies to 1-3 sentences, and end with a question to keep the conversation going. "
    "The user's messages are speech-to-text transcripts, so ignore punctuation/capitalization issues.\n\n"
    "Respond in EXACTLY this format:\n"
    "<your English reply>\n" + SEP + "\n"
    "<feedback in Japanese, max 4 bullet points: (1) 文法・語彙の誤りと自然な言い換え "
    "(2) より良い表現の提案 (3) 発音データがあれば、特に直すべき点を1つ。"
    "誤りがなければ短く褒める>"
)


def transcribe(wav_path: str) -> str:
    """Whisper で音声を英語テキストに変換する(遅延 import)。"""
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if "whisper" not in _cache:
        _cache["wp"] = WhisperProcessor.from_pretrained(WHISPER_ID)
        _cache["wm"] = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID).eval()
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    feats = _cache["wp"](audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = _cache["wm"].generate(feats)
    return _cache["wp"].batch_decode(ids, skip_special_tokens=True)[0].strip()


def parse_reply(raw: str) -> tuple[str, str]:
    """Claude の出力を (英語の返答, 日本語フィードバック) に分割する。"""
    if SEP in raw:
        reply, fb = raw.split(SEP, 1)
        return reply.strip(), fb.strip()
    return raw.strip(), ""


def chat_turn(
    history: list[dict],
    user_text: str | None,
    topic: str,
    level: str,
    pron_note: str = "",
) -> tuple[str, str]:
    """history は [{"role","content"(str)}...]。user_text が None なら会話の開始(Claude から話しかける)。"""
    messages = list(history)
    if user_text is None:
        messages.append({"role": "user", "content": "(Start the conversation with a short greeting and your first question.)"})
    else:
        content = user_text if not pron_note else f"{user_text}\n\n[Pronunciation data: {pron_note}]"
        messages.append({"role": "user", "content": content})

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},  # 会話の応答速度を優先
        system=SYSTEM_TMPL.format(topic=topic, level=LEVELS.get(level, LEVELS["中級"])),
        messages=messages,
    )
    if resp.stop_reason == "refusal":
        return "Sorry, I can't continue with that. Let's talk about something else.", ""
    raw = "".join(b.text for b in resp.content if b.type == "text")
    return parse_reply(raw)
