"""音素列の比較による発音スコア計算(外部依存なし)。"""
from __future__ import annotations

from dataclasses import dataclass

_IGNORED = set("ˈˌːˑ‿ ")  # ストレス・長音記号・空白は比較から除外


def normalize(phonemes: list[str]) -> list[str]:
    """ストレス/長音記号を取り除き、空になった音素を捨てる。"""
    out = []
    for p in phonemes:
        p = "".join(c for c in p if c not in _IGNORED)
        if p:
            out.append(p)
    return out


@dataclass
class PhonemeResult:
    expected: str | None  # 期待された音素(余分に発音された場合は None)
    heard: str | None  # 認識された音素(脱落した場合は None)
    status: str  # "ok" | "substituted" | "missing" | "extra"


def align(expected: list[str], heard: list[str]) -> list[PhonemeResult]:
    """Needleman-Wunsch で期待音素列と認識音素列をアライメントする。"""
    e, h = normalize(expected), normalize(heard)
    n, m = len(e), len(h)
    # コスト: 一致0 / 置換1 / 挿入・脱落1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (0 if e[i - 1] == h[j - 1] else 1)
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    results: list[PhonemeResult] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if e[i - 1] == h[j - 1] else 1):
            same = e[i - 1] == h[j - 1]
            results.append(PhonemeResult(e[i - 1], h[j - 1], "ok" if same else "substituted"))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            results.append(PhonemeResult(e[i - 1], None, "missing"))
            i -= 1
        else:
            results.append(PhonemeResult(None, h[j - 1], "extra"))
            j -= 1
    results.reverse()
    return results


def score(results: list[PhonemeResult]) -> float:
    """0〜100 のスコア。正しい音素数 / 期待音素数(余分な音素は減点)。"""
    expected_n = sum(1 for r in results if r.expected is not None)
    if expected_n == 0:
        return 0.0
    ok = sum(1 for r in results if r.status == "ok")
    extra = sum(1 for r in results if r.status == "extra")
    return round(max(0.0, (ok - 0.5 * extra) / expected_n) * 100, 1)


def mistakes(results: list[PhonemeResult]) -> list[str]:
    """人間/LLM が読める形の誤りリスト。"""
    msgs = []
    for r in results:
        if r.status == "substituted":
            msgs.append(f"/{r.expected}/ が /{r.heard}/ に聞こえた")
        elif r.status == "missing":
            msgs.append(f"/{r.expected}/ が抜けていた")
        elif r.status == "extra":
            msgs.append(f"余分な /{r.heard}/ が入っていた")
    return msgs
