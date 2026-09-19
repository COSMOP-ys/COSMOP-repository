# jev_voice_cv — voice operation with CV grounding and Jev typed decisions

A prototype of the pipeline the voice-accuracy question came down to:

```
speech ─▶ ASR ─▶ Jev: is this addressed to the computer?   (yes/no, calibrated)
                 │
                 ├─▶ Jev: which action?                    (one-of-N, closed set)
                 │
                 ├─▶ SAM 3.1: which regions match the words? (boxes + masks)
                 │     └─▶ Jev: which candidate?           (one-of-N over those regions)
                 │
                 └─▶ gate: execute / speculate / confirm / reject / hold
```

## What this changes about accuracy, and what it does not

| Layer | Effect |
|---|---|
| Speech recognition | **Unchanged.** Jev takes no audio; your ASR is still your ASR |
| Intent → action | **Better.** The answer cannot leave the enumerated set, so no invented action names and nothing to parse |
| Utterance → on-screen target | **Better, and this is the new part.** SAM 3.1 turns "the red submit button" into concrete candidates; Jev picks one of them instead of a coordinate being guessed |
| Free-form arguments (note body, search string) | **Not covered.** Jev generates no text; `needs_text_arg` routes that to a small LLM |
| Wrong action actually firing | **Better, if you use the probabilities.** Per-action thresholds, destructive actions always confirming, and read-only-only speculation are what buy this — not the model by itself |

## Run it

```sh
python3 -m jev_voice_cv.cli          # offline demo, deterministic fake model, no keys
python3 -m unittest discover -s tests -t .
```

Stdlib only, no install step. The demo streams partial transcripts through the
pipeline and prints the gate decision per revision:

```
  [partial] 'open the notes app'                    -> speculate  read-only and confident on partial
  [final  ] 'open the notes app and create a ...'   -> execute    final transcript, above execute threshold
            skip (already ran): open_app
  [partial] 'delete the red'                        -> hold       partial transcript, action is not read-only+idempotent
  [final  ] 'delete the red delete button'          -> confirm    action always requires confirmation
  [final  ] 'so anyway I told him to open a ticket' -> reject     not addressed to the computer
```

## The four rules doing the work

1. **Per-action thresholds** (`ActionSpec`). One global cutoff either blocks
   harmless reads or lets `delete` through on a mishearing. Scale the bar to the
   cost of being wrong; ~0.85 is a starting point to tune on your own data.
2. **Speculation allowlist.** Acting before the sentence ends is where the speed
   comes from, so partials may only trigger `read_only and idempotent` actions,
   and only above a bar strictly higher than the end-of-utterance one. Nothing
   speculative ever needs undoing.
3. **Supersede and discard.** Each transcript revision bumps a sequence number;
   a decision that returns for an older ticket comes back `SUPERSEDED` and is
   never executed (`VoicePipeline.resolve` re-checks after every model call).
4. **No target, no action.** If CV grounds nothing, the pipeline asks or waits.
   It never falls through to a guessed coordinate.

`policy.combine` multiplies the intent and target probabilities. That assumes
independence, which is false — Jev reads the same transcript twice, so correlated
errors make the product optimistic. Good enough as a gate, not as a calibration
claim.

## Wiring the real APIs

| | value (checked 2026-09-19) |
|---|---|
| Jev model id | `~typesafe/jev-latest`, pinned `typesafe/jev-1.13` |
| Jev price / latency | $0.042 per 1M input tokens, output free; 70–500 ms; 32k context; OpenRouter **beta** |
| SAM 3.1 price | $2.50 / 1k images, $0.20 / 1k video frames; up to 16 tracked targets (Object Multiplex) |
| Keys | `OPENROUTER_API_KEY`, `META_API_KEY` |

**The two transports are unverified.** openrouter.ai and developer.meta.com are
both blocked from the sandbox this was written in, so the request and response
mapping was never exercised against the live beta. Check these before the first
real call — everything else is independent of them:

- `jev.py`: `DEFAULT_ENDPOINT`, `_post` body, `_parse_probabilities`
- `grounding.py`: `DEFAULT_ENDPOINT`, `ground` body, `_parse_candidates`

`DryRunExecutor` is the default on purpose. Keep it until the thresholds are
tuned against recorded sessions.

## What to measure

Measure these three separately, or ASR failures get blamed on Jev and vice versa:

- **WER** on your own audio — the speech layer, unaffected by anything here.
- **Intent accuracy** and **target accuracy** on transcripts, held fixed.
- **False-local-action rate**: actions executed that the speaker did not want.
  This is the number the thresholds are tuned against.

Also check calibration directly: bucket decisions by reported probability and
compare to the observed hit rate. A model that says 0.70 should be right about
70% of the time; TypeSafe publishes no calibration guarantee, so verify on your
data before trusting a threshold.

## Data leaving the machine

Every grounded turn ships a screenshot to Meta's API and the app state to
OpenRouter/TypeSafe. On a real desktop that can include mail, tokens and
customer data. Options: crop the frame to the active window, ground from the
accessibility tree or DOM instead of pixels (`StaticGrounder` is the same
interface and costs nothing), or gate grounding behind push-to-talk.

## Known limits

- **One action per utterance.** "Open notes *and* create a list" resolves to
  `open_app` only; chaining needs a sequence decision or a planner above Jev.
- **No free-text extraction.** `needs_text_arg` marks where a small LLM goes.
- **Synchronous by design.** `submit`/`resolve` are split so the staleness rules
  survive a move to async, but there is no event loop here yet.
- **`ActionSpec` set is an example**, not a recommendation. Yours should come from
  the actions your app actually exposes.

## Related work (not audited)

Same shape, independently: `browser-use/jev-ultrafast` (Jev for element choice,
small LLM only for typing), `moritzkremb/jev-voice-browser` (browser speech →
Playwright), `awlevin/typesafe-computer-use` (macOS OCR + accessibility),
`realZachi/pg-jev` and `superagents-lab/jev-search` (Jev as a ranking/filter
stage). Listed for orientation — none of them were read while writing this.
