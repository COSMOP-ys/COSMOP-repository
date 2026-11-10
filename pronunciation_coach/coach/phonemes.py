"""期待音素の生成(espeak-ng)と、音声からの音素認識(wav2vec2)。重い依存は遅延 import。"""
from __future__ import annotations

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
_cache: dict = {}


def expected_phonemes(text: str, language: str = "en-us") -> list[str]:
    """テキストから期待される音素列(espeak-ng の音素)を返す。"""
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    out = phonemize(
        text,
        language=language,
        backend="espeak",
        separator=Separator(phone=" ", word=" | ", syllable=""),
        strip=True,
    )
    return [p for p in out.split() if p != "|"]


def recognize_phonemes(wav_path: str) -> list[str]:
    """音声ファイルから音素列を認識して返す(16kHz mono に変換)。"""
    import librosa
    import torch
    from transformers import AutoModelForCTC, Wav2Vec2Processor

    if "model" not in _cache:
        _cache["processor"] = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _cache["model"] = AutoModelForCTC.from_pretrained(MODEL_ID).eval()
    processor, model = _cache["processor"], _cache["model"]

    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].split()


def record(seconds: float, out_path: str) -> str:
    """マイクから録音して WAV 保存する。"""
    import sounddevice as sd
    import soundfile as sf

    print(f"{seconds:.0f} 秒間録音します。話してください…")
    data = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    sf.write(out_path, data, 16000)
    return out_path
