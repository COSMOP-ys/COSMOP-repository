# jev_voice_cv — voice operation with on-screen grounding and Jev typed decisions

A prototype of the pipeline the voice-accuracy question came down to. Speech
recognition gets you the words; the hard half is turning "click the red one"
into a specific element, and then deciding whether you are sure enough to act.

```
speech ─▶ ASR ─▶ Jev, one request, two questions:
                 │   • is this addressed to the computer?   (boolean, probability)
                 │   • which action?                        (one-of-N, closed set)
                 │
                 ├─▶ grounding: which things on screen match those words?
                 │     DOM accessibility tree (default, local, free)
                 │     SAM 3.1                (fallback: canvas, video, native)
                 │     └─▶ Jev: which candidate?            (one-of-N over those)
                 │
                 └─▶ gate: execute / speculate / confirm / reject / hold
```

## What this changes about accuracy, and what it does not

| Layer | Effect |
|---|---|
| Speech recognition | **Unchanged.** Jev takes no audio; your ASR is still your ASR |
| Intent → action | **Better.** The answer cannot leave the enumerated set, so no invented action names and nothing to parse |
| Utterance → on-screen target | **Better, and this is the new part.** The DOM (or SAM 3.1) turns "the red submit button" into concrete candidates; Jev picks one of them instead of a coordinate being guessed |
| Free-form arguments (note body, search string) | **Not covered.** Jev generates no text; `needs_text_arg` routes that to a small LLM, and `plan.with_text(...)` carries the result |
| Wrong action actually firing | **Better, if you use the probabilities.** Per-action thresholds, destructive actions always confirming, and read-only-only speculation are what buy this — not the model by itself |

## Run it

```sh
python -m jev_voice_cv.cli                       # offline demo, no keys
python -m jev_voice_cv.web.server --offline      # voice console on :8765, no keys
python -m unittest discover -s tests -t .        # 92 tests, stdlib only
```

The package is stdlib-only. Playwright is needed for `playwright_exec` and the
browser tests, which skip cleanly when it is absent:

```sh
python -m venv .venv && .venv/Scripts/python -m pip install playwright
.venv/Scripts/python -m unittest tests.test_browser   # 13 tests, real browser
```

### The voice console

`web/server.py` serves a page that streams every interim speech result — or
every keystroke, which takes the identical path and needs no microphone —
together with a snapshot of its own accessibility tree. It shows the verdict per
revision and applies the ones the gate cleared. With `--offline` it runs a
keyword stub instead of Jev, so the whole loop works with no key and no network.

## The five rules doing the work

1. **Per-action thresholds** (`ActionSpec`). One global cutoff either blocks
   harmless reads or lets `delete` through on a mishearing. Scale the bar to the
   cost of being wrong; ~0.85 is a starting point to tune on your own data.
2. **Speculation allowlist.** Acting before the sentence ends is where the speed
   comes from, so partials may only trigger `read_only and idempotent` actions,
   and only above a bar strictly higher than the end-of-utterance one. Nothing
   speculative ever needs undoing.
3. **Supersede and discard.** Each transcript revision bumps a sequence number;
   a decision that returns for an older ticket comes back `SUPERSEDED` and is
   never executed (`resolve` re-checks after every model call).
4. **No target, no action.** If grounding finds nothing, the pipeline asks or
   waits. It never falls through to a guessed coordinate.
5. **Act on the ref, not the description.** The snapshot stamps `data-jev-ref`
   on each element and the executor addresses that. Between grounding and
   execution the page can move, and re-matching "the second red button" would
   hit whatever took its place.

`policy.combine` multiplies the intent and target probabilities. That assumes
independence, which is false — Jev reads the same transcript twice, so correlated
errors make the product optimistic. Good enough as a gate, not as a calibration
claim.

## Grounding: DOM first, pixels only when you must

`DomGrounder` reads the accessibility tree: names, roles, placeholders, and the
computed colour of every control, so "the red delete button" matches on all
three words without a screenshot existing. It is local, free, and nothing leaves
the machine. `Sam31Grounder` is the fallback for what the DOM cannot describe —
`<canvas>`, video, native windows.

Either way the grounder is a **recall** device: it shortlists cheaply and Jev
makes the one-of-N pick. Its `score` is a lexical heuristic — weighted by how
much each spoken word narrows the page down — deliberately never 1.0, and worth
recalibrating against your own logs. A word that belongs to one element scores
high; a word every button shares does not.

## Wiring the real APIs

| | value (checked 2026-09-19) |
|---|---|
| Jev via Vercel AI Gateway | model `typesafe-ai/jev`, 32k context, `POST https://ai-gateway.vercel.sh/v4/ai/evaluation-model` |
| Jev price | $0.042 / 1M input, output free — currently $0 both ways under promotional pricing **until 2026-09-25** |
| Jev free tier | Free Tier eligible, so the $5/month included credit covers it. Buying credits moves the team to the paid tier and **permanently** ends that monthly credit |
| SAM 3.1 price | $2.50 / 1k images, $0.20 / 1k video frames; up to 16 tracked targets (Object Multiplex) |
| Keys | `AI_GATEWAY_API_KEY`, `META_API_KEY` |

**On the Jev transport.** Vercel documents evaluation as "available through the
AI SDK only" — there is no published REST endpoint. The request in `jev.py` was
derived from the AI SDK's own gateway provider (`@ai-sdk/gateway@4.0.87`,
`getUrl()` → `${baseURL}/evaluation-model`, body `{state, questions}`, headers
`ai-evaluation-model-specification-version: 4` and `ai-model-id`) and confirmed
against the live service: unauthenticated it answers 401 `authentication_error`,
and one character off the path it answers 404. Undocumented means it can move
without notice — `DEFAULT_ENDPOINT` and `_parse_answer` are the two places to
re-check. Nothing in `policy.py` or `pipeline.py` depends on either.

**The SAM transport is still unverified**: developer.meta.com was unreachable
when this was written, so check `grounding.py`'s `DEFAULT_ENDPOINT`, request
body and `_parse_candidates` before the first real call. The DOM path does not
need it.

`DryRunExecutor` is the default on purpose. Keep it until the thresholds are
tuned against recorded sessions.

## Round trips

Two per revision, not three. "Is this addressed to the computer?" and "which
action?" are different questions about the same state, and Jev answers several
in parallel in one request — so the filter question costs a few input tokens and
no latency. The target question has to be separate: its option set does not
exist until an action that needs a target has been chosen.

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

With DOM grounding, what leaves is the transcript plus element names and roles —
no pixels. With `Sam31Grounder`, every grounded turn ships a screenshot to
Meta's API, which on a real desktop can include mail, tokens and customer data.
Options: crop the frame to the active window, stay on the DOM path, or gate
grounding behind push-to-talk. AI Gateway reports Jev as zero-data-retention and
no-training; that covers the provider path, not your own logs.

## Known limits

- **One action per utterance.** "Open notes *and* create a list" resolves to
  `open_app` only; chaining needs a sequence decision or a planner above Jev.
- **No free-text extraction.** `needs_text_arg` marks where a small LLM goes.
- **Synchronous by design.** `submit`/`resolve` are split so the staleness rules
  survive a move to async, but there is no event loop here yet.
- **Web Speech API** is Chrome/Edge only and sends audio to their speech
  service. The console's text box exercises the same path without it.
- **`ActionSpec` set is an example**, not a recommendation. Yours should come from
  the actions your app actually exposes.

## Related work (not audited)

Same shape, independently: `browser-use/jev-ultrafast` (Jev for element choice,
small LLM only for typing), `moritzkremb/jev-voice-browser` (browser speech →
Playwright), `awlevin/typesafe-computer-use` (macOS OCR + accessibility),
`realZachi/pg-jev` and `superagents-lab/jev-search` (Jev as a ranking/filter
stage). Listed for orientation — none of them were read while writing this.
