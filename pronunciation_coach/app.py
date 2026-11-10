"""Web UI: python app.py  →  http://127.0.0.1:7860"""
from __future__ import annotations

import html
import os

import gradio as gr

from coach.pipeline import assess

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


with gr.Blocks(title="Pronunciation Coach") as demo:
    gr.Markdown("# 🎤 English Pronunciation Coach\n英文を読み上げて録音 → 音素レベルで採点 → Claude が日本語でアドバイス")
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

if __name__ == "__main__":
    demo.launch()
