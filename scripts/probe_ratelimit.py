"""How long is the free tier's rate-limit window, measured rather than guessed.

Vercel publishes no numbers ("limits can change, so this page describes
behavior rather than fixed numbers"). This polls one minimal request until it
is served again and reports how long that took, then measures how many calls
in a row the tier will actually take.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jev_voice_cv.jev import DEFAULT_ENDPOINT, DEFAULT_MODEL  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe_jev import _key_from_env_file  # noqa: E402

KEY = _key_from_env_file()
BODY = json.dumps({
    "state": "ping",
    "questions": {"q": {"type": "boolean", "instructions": "Is this a ping?"}},
}).encode()


def call() -> tuple[int, float]:
    request = urllib.request.Request(
        DEFAULT_ENDPOINT, data=BODY,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "ai-evaluation-model-specification-version": "4",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-model-id": DEFAULT_MODEL,
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
            return 200, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        exc.read()
        return exc.code, (time.perf_counter() - started) * 1000


def main() -> int:
    t0 = time.time()
    print("waiting for the limit to clear (polling every 20s, giving up at 15 min)", flush=True)
    while True:
        status, _ = call()
        waited = time.time() - t0
        if status == 200:
            print(f"cleared after ~{waited:.0f}s", flush=True)
            break
        if waited > 900:
            print(f"still limited after {waited:.0f}s - the window is longer than 15 min", flush=True)
            return 1
        time.sleep(20)

    print("\nnow: how many back-to-back calls does it take before 429?", flush=True)
    latencies = []
    for i in range(1, 31):
        status, ms = call()
        if status != 200:
            print(f"  429 on call {i}  (after {i - 1} served)", flush=True)
            break
        latencies.append(ms)
        print(f"  {i:2d}: 200  {ms:.0f} ms", flush=True)
        time.sleep(1.0)
    if latencies:
        latencies.sort()
        print(f"\nlatency p50 {latencies[len(latencies) // 2]:.0f} ms  "
              f"min {latencies[0]:.0f}  max {latencies[-1]:.0f}  n={len(latencies)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
