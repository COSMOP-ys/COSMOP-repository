"""返答の読み上げ(ローカル TTS: VITS)。重い依存は遅延 import。"""
from __future__ import annotations

import os
import tempfile

TTS_ID = os.environ.get("TTS_MODEL", "facebook/mms-tts-eng")
_cache: dict = {}


def synthesize(text: str) -> str:
    """英文を WAV に合成してパスを返す。"""
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    if "model" not in _cache:
        _cache["tok"] = AutoTokenizer.from_pretrained(TTS_ID)
        _cache["model"] = VitsModel.from_pretrained(TTS_ID).eval()
    model, tok = _cache["model"], _cache["tok"]
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        wave = model(**inputs).waveform[0].cpu().numpy()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, wave, model.config.sampling_rate)
    return path
