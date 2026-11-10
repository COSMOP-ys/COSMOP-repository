"""CLI と Web UI で共通の採点パイプライン。"""
from __future__ import annotations

from dataclasses import dataclass

from .scoring import PhonemeResult, align, mistakes, score


@dataclass
class Assessment:
    expected: list[str]
    heard: list[str]
    results: list[PhonemeResult]
    score: float
    mistakes: list[str]


def assess(text: str, wav_path: str) -> Assessment:
    from .phonemes import expected_phonemes, recognize_phonemes

    expected = expected_phonemes(text)
    heard = recognize_phonemes(wav_path)
    results = align(expected, heard)
    return Assessment(expected, heard, results, score(results), mistakes(results))
