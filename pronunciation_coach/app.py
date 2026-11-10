"""Web UI: python app.py  →  http://127.0.0.1:7860"""
from __future__ import annotations

import html
import os

import gradio as gr

from coach.pipeline import assess
from coach import conversation as conv
from coach import tts

SAMPLES = [
    "I think the right answer is light.",
    "She sells seashells by the seashore.",
    "Could I have a glass of very cold water, please?",
    "The weather is rather nice this Thursday.",
]
COLORS = {"ok": "#2e7d32", "substituted": "#e65100", "missing": "#c62828", "extra": "#6a1b9a"}


def _render(results) -> str:
    """音素ごとに色分けした HTML(緑=正解 / 橙=置換 / 赤=脱落 / 紫=余分)。"""
    cells = []
    for r in results:
        label = html.escape(r.heard or r.expected or "")
        tip = {
            "ok": "OK",
            "substituted": f"期待 /{r.expected}/ → 認識 /{r.heard}/",
            "missing": f"/{r.expected}/ が抜けている",
            "extra": f"余分な /{r.heard}/",
        }[r.status]
        style = f"color:#fff;background:{COLORS[r.status]};"
        if r.status == "missing":
            label = html.escape(r.expected or "")
            style += "text-decoration:line-through;"
        cells.append(
            f'<span title="{html.escape(tip)}" style="{style}padding:2px 8px;margin:2px;'
            f'border-radius:6px;display:inline-block;font-size:1.1em">{label}</span>'
        )
    legend = (
        '<div style="margin-top:8px;font-size:.85em;opacity:.8">'
        "緑=正解 / 橙=置換 / 赤(取消線)=脱落 / 紫=余分</div>"
    )
    return "<div>" + "".join(cells) + "</div>" + legend


def run(text: str, audio_path: str | None, use_claude: bool):
    if not text.strip():
        raise gr.Error("練習する英文を入力してください。")
    if not audio_path:
        raise gr.Error("音声を録音するか、ファイルをアップロードしてください。")
    try:
        a = assess(text, audio_path)
    except Exception as e:  # モデル未取得・espeak 未インストールなどを画面に出す
        raise gr.Error(f"採点に失敗しました: {e}")

    advice = ""
    if use_claude:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            advice = "(ANTHROPIC_API_KEY が未設定のため、Claude のアドバイスは省略しました)"
        else:
            try:
                from coach.feedback import coach

                advice = coach(text, a.score, a.mistakes)
            except Exception as e:
                advice = f"(Claude の呼び出しに失敗しました: {e})"
    return f"{a.score} / 100", _render(a.results), "\n".join(f"- {m}" for m in a.mistakes) or "誤りなし", advice


TOPICS = ["カフェで注文する", "空港・ホテルでのやりとり", "仕事の面接", "自己紹介・雑談", "道を尋ねる"]


def _need_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise gr.Error("会話モードには ANTHROPIC_API_KEY の設定が必要です。")


def _speak(text, enabled):
    if not enabled:
        return None
    try:
        return tts.synthesize(text)
    except Exception as e:
        gr.Warning(f"読み上げに失敗しました: {e}")
        return None


def start_chat(topic, level, speak):
    _need_key()
    reply, fb = conv.chat_turn([], None, topic, level)
    hist = [{"role": "user", "content": "(start)"}, {"role": "assistant", "content": reply}]
    return hist, [{"role": "assistant", "content": reply}], "会話を始めました。録音して返事をしてみましょう。", _speak(reply, speak)


def talk(audio_path, hist, shown, topic, level, do_pron, speak):
    _need_key()
    if not hist:
        raise gr.Error("先に「会話を始める」を押してください。")
    if not audio_path:
        raise gr.Error("音声を録音してください。")
    try:
        text = conv.transcribe(audio_path)
    except Exception as e:
        raise gr.Error(f"文字起こしに失敗しました: {e}")
    if not text:
        raise gr.Error("音声を認識できませんでした。もう一度録音してください。")

    note, score_txt = "", ""
    if do_pron:
        try:
            a = assess(text, audio_path)
            note = f"score {a.score}/100; " + ("; ".join(a.mistakes[:6]) or "no errors detected")
            score_txt = f"**発音スコア: {a.score}/100**\n\n" + ("\n".join(f"- {m}" for m in a.mistakes[:6]) or "誤りなし")
        except Exception as e:
            score_txt = f"(発音採点をスキップ: {e})"
    reply, fb = conv.chat_turn(hist, text, topic, level, note)
    hist = hist + [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    shown = shown + [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    return hist, shown, None, score_txt + ("\n\n---\n" if score_txt else "") + fb, _speak(reply, speak)


with gr.Blocks(title="Pronunciation Coach") as demo:
    gr.Markdown("# 🎤 English Pronunciation Coach")
    with gr.Tab("発音練習"):
        gr.Markdown("英文を読み上げて録音 → 音素レベルで採点 → Claude が日本語でアドバイス")
        with gr.Row():
            with gr.Column():
                sample = gr.Dropdown(SAMPLES, label="サンプル文", value=SAMPLES[0])
                text = gr.Textbox(label="練習する英文", value=SAMPLES[0], lines=2)
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="音声")
                use_claude = gr.Checkbox(value=True, label="Claude のアドバイスを表示")
                btn = gr.Button("採点する", variant="primary")
            with gr.Column():
                score_box = gr.Textbox(label="スコア", interactive=False)
                phones = gr.HTML(label="音素ごとの結果")
                errs = gr.Markdown(label="検出された誤り")
                advice = gr.Markdown(label="コーチからのアドバイス")
        sample.change(lambda s: s, sample, text)
        btn.click(run, [text, audio, use_claude], [score_box, phones, errs, advice])

    with gr.Tab("会話練習"):
        gr.Markdown("Claude と英語で会話 → 毎ターン、文法の添削と発音スコアを表示します。")
        hist_state = gr.State([])  # Claude に渡す履歴
        with gr.Row():
            topic = gr.Dropdown(TOPICS, value=TOPICS[0], label="シチュエーション", allow_custom_value=True)
            level = gr.Radio(list(conv.LEVELS), value="中級", label="レベル")
            do_pron = gr.Checkbox(value=True, label="発音も採点する")
            speak = gr.Checkbox(value=True, label="返答を読み上げる")
            start_btn = gr.Button("会話を始める / リセット", variant="primary")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="会話", height=380)
                reply_audio = gr.Audio(label="返答の音声", autoplay=True, interactive=False, type="filepath")
                chat_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="あなたの返事(録音)")
                talk_btn = gr.Button("送信")
            with gr.Column(scale=2):
                feedback = gr.Markdown("ここに添削と発音スコアが表示されます。")
        start_btn.click(start_chat, [topic, level, speak], [hist_state, chat, feedback, reply_audio])
        talk_btn.click(talk, [chat_audio, hist_state, chat, topic, level, do_pron, speak],
                       [hist_state, chat, chat_audio, feedback, reply_audio])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
