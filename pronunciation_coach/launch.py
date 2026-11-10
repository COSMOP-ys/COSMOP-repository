"""ワンクリック起動: 仮想環境の作成 → 依存インストール → アプリ起動 → ブラウザを開く。

  python launch.py            # 通常起動
  python launch.py --dry-run  # 何をするかだけ表示
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ = ROOT / "requirements.txt"
STAMP = VENV / ".requirements.sha256"
ENV_FILE = ROOT / ".env"
DRY = "--dry-run" in sys.argv


def say(msg: str) -> None:
    print(f"[launcher] {msg}", flush=True)


def load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def ensure_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    say("ANTHROPIC_API_KEY が未設定です(会話練習と Claude のアドバイスに必要)。")
    if DRY or not sys.stdin.isatty():
        return
    key = input("API キーを貼り付けて Enter(スキップは空 Enter): ").strip()
    if key:
        ENV_FILE.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
        os.environ["ANTHROPIC_API_KEY"] = key
        say(f"{ENV_FILE.name} に保存しました(Git には含まれません)。")


def main() -> int:
    say(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        say("Python 3.10 以上が必要です。")
        return 1

    if not shutil.which("espeak-ng"):
        say("⚠ espeak-ng が見つかりません。発音採点に必要です。")
        say("  macOS: brew install espeak-ng / Ubuntu: sudo apt-get install espeak-ng")
        say("  Windows: https://github.com/espeak-ng/espeak-ng/releases")

    if not PY.exists():
        say("仮想環境を作成します(初回のみ)…")
        if not DRY:
            venv.create(VENV, with_pip=True)

    digest = hashlib.sha256(REQ.read_bytes()).hexdigest()
    if not STAMP.exists() or STAMP.read_text().strip() != digest:
        say("依存ライブラリをインストールします(初回は数分かかります)…")
        if not DRY:
            subprocess.check_call([str(PY), "-m", "pip", "install", "-r", str(REQ)])
            STAMP.write_text(digest)

    load_env()
    ensure_api_key()

    say("アプリを起動します。ブラウザが自動で開きます(終了は Ctrl+C またはこの画面を閉じる)。")
    if DRY:
        return 0
    return subprocess.call([str(PY), str(ROOT / "app.py")], cwd=ROOT)


if __name__ == "__main__":
    try:
        code = main()
    except subprocess.CalledProcessError as e:
        say(f"失敗しました: {e}")
        code = 1
    if code and os.name == "nt":
        input("Enter キーで閉じます…")
    sys.exit(code)
