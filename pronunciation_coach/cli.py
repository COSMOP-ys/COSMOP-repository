"""使い方:
  python cli.py "I think the right answer is light." --wav my.wav
  python cli.py "I think the right answer is light." --record 5
  python cli.py "..." --wav my.wav --no-claude   # Claude を使わずスコアのみ
"""
from __future__ import annotations

import argparse
import tempfile

from coach.pipeline import assess


def main() -> None:
    ap = argparse.ArgumentParser(description="英語発音採点 + Claude コーチング")
    ap.add_argument("text", help="読み上げる英文")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", help="録音済み WAV ファイル")
    g.add_argument("--record", type=float, metavar="SEC", help="マイクで N 秒録音")
    ap.add_argument("--no-claude", action="store_true", help="Claude のフィードバックを省略")
    args = ap.parse_args()

    from coach.phonemes import record

    wav = args.wav or record(args.record, tempfile.mkstemp(suffix=".wav")[1])
    a = assess(args.text, wav)
    s, errs = a.score, a.mistakes

    print(f"期待: {' '.join(a.expected)}\n認識: {' '.join(a.heard)}")
    print(f"\nスコア: {s}/100")
    for e in errs:
        print(f"  - {e}")

    if not args.no_claude:
        from coach.feedback import coach

        print("\n--- コーチからのアドバイス ---")
        print(coach(args.text, s, errs))


if __name__ == "__main__":
    main()
