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
python -m unittest discover -s tests -t .        # 99 tests, stdlib only
```

With a key in `.env.local` (gitignored), two probes answer the questions that
only a live call can:

```sh
python scripts/probe_jev.py --batched   # wire format, latency, calibration table
python scripts/probe_ratelimit.py       # how many calls the tier actually serves
```

`--batched` folds the whole labelled set into one request, because the free
tier will not serve 24. It is a shape check, not the request the pipeline
sends: see the docstring for why the two differ.

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

## Languages

Jev needs nothing done to it. Measured 2026-09-19 on Japanese transcripts
against these same English action descriptions and question wordings — no
translation anywhere — intent was right on every clear command, the aside
「さっき彼にチケットを開いてって言ったんだよ」 scored 0.06 on *addressed*, and
clarity separated exactly as it does in English (commands 3.5–3.8, non-commands
0.05–0.21).

The grounder is where the work was. A `[a-z0-9]+` tokenizer returns **zero**
tokens for Japanese — not a poor match, no candidates at all — and Japanese
writes no spaces, so widening the character class does not help either.
`dom.py` splits on script boundaries instead, which is a decent segmentation
heuristic because kanji/kana alternation marks word edges:

```
変更を保存ボタンを押して  ->  変更 | を | 保存 | ボタン | を | 押 | して
                          content   particle    content   particle   verb
```

The grammatical hiragana runs drop out as stopwords, character bigrams let
メール reach メールアドレス, NFKC folds full-width and half-width forms
together, and a synonym table maps spoken words into the vocabulary the DOM
actually emits (赤 → `red`, ボタン → `button`), since `getComputedStyle` names
colours in English whatever the page is written in.

Verified end to end against a real browser and a Japanese page: snapshot →
shortlist → gate → real click, plus 「アカウントを削除して」 stopping at
CONFIRM. The Web Speech API side is a `lang` selector in the console, defaulted
from `navigator.language`.

This generalises to Chinese and Korean about as far as the heuristic does: Han
runs and bigrams are the same machinery, but the stopword and synonym lists are
Japanese, so treat those as the part to extend.

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
`ai-evaluation-model-specification-version: 4` and `ai-model-id`), and **it
works** — 200 against the live service, verified 2026-09-19:

```json
{"answers": {"intent":    {"type": "choice", "choice": "click_element",
                           "probabilities": {"click_element": 1, "open_app": 0, ...}},
             "addressed": {"type": "boolean", "probability": 0.64}},
 "rounding": {"probabilityDecimals": 2, "scoreDecimals": 2},
 "usage":    {"inputTokens": 553, "outputTokens": 109}}
```

`choice` returns the full distribution `policy.py` gates on, probabilities come
back rounded to two decimals, and `providerMetadata.gateway.cost` was `"0"`
against a `marketCost` of `$0.0000232` for that call. Undocumented still means
it can move without notice — `DEFAULT_ENDPOINT` and `_parse_answer` are the two
places to re-check. Nothing in `policy.py` or `pipeline.py` depends on either.

**Getting a key costs nothing but is not frictionless.** AI Gateway refuses
every request with 403 `customer_verification_required` until a card is on
file, free credits included. Adding the card is enough; buying credits ends the
monthly free credit for good.

**"Free" is about money, not about throughput.** Vercel publishes no rate-limit
numbers ("this page describes behavior rather than fixed numbers"), so
`scripts/probe_ratelimit.py` measures them. On the free tier, 2026-09-19:

```
 4 calls served back to back, 429 on the 5th
 ~165 s before the limit cleared
 p50 413 ms per call (min 379, max 509)
```

Roughly **four requests every three minutes**. That is fine for trying it out
and useless for anything that decides per utterance — this pipeline sends one
request per transcript revision, several per second while someone is speaking.
Any real workload needs purchased credits, which is the same switch that ends
the $5/month free credit. Budget for the paid tier or do not plan around Jev.

The 413 ms is worth noting on its own: it sits at the slow end of the advertised
70–500 ms, and it is the round trip the speculation design spends on every
partial transcript.

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

## What the probabilities actually look like

24 labelled utterances, 2026-09-19, run twice: once per utterance (the request
the pipeline actually sends) and once batched into a single request. Small, one
sample each — but the two agree on the shape, so it is not an artifact of
batching.

```
                one request per utterance          all 24 in one request
intent          83% correct (20/24)                88% correct (21/24)
  reported     n   said  right                reported     n   said  right
  0.95-1.00   18   1.00   0.89                0.95-1.00   21   1.00   0.90
  0.85-0.95    2   0.92   1.00                0.85-0.95    1   0.91   1.00
  0.70-0.85    2   0.77   0.50                0.50-0.70    1   0.68   0.00
  0.50-0.70    2   0.59   0.50

addressed       88% correct                        92% correct
  0.95-1.00    2   0.96   1.00                0.95-1.00    2   0.96   1.00
  0.85-0.95    5   0.88   0.80                0.85-0.95   10   0.90   1.00
  0.70-0.85    6   0.81   1.00                0.70-0.85   11   0.80   0.91
  0.50-0.70   11   0.61   0.82

latency p50 450 ms (min 347), n=24
```

**The intent axis is saturated.** Eighteen of twenty-four answers report 1.00,
and one in nine of those is wrong. Every `execute_threshold` between 0.70 and
0.99 therefore admits the same set — on the intent axis the dial is close to a
no-op, and a wrong answer arrives wearing the same 1.00 as a right one.

**The boolean is the axis with real gradation, and it sits low.** Per utterance,
*eleven of twenty-four* land in 0.50–0.70, averaging 0.61 — against a default
`command_threshold` of 0.60. Nearly half the traffic decides within a hundredth
of the line, which means whether a perfectly ordinary command is heard at all
comes down to rounding. Raise the bar and real commands get dropped; lower it
and the filter stops filtering. It is 82% right in that band, so the ordering
carries information — the threshold is just in the worst possible place.
Batching hid this: with all 24 transcripts in view the same question came back
around 0.80.

What this says about the design: the safety comes from the structure, not from
the number. Look at what caught the two dangerous misses.

- `"she said we should delete the whole thing"` → `delete_element` at **1.00**,
  comfortably past delete's 0.95 bar. Stopped by `always_confirm`, and by the
  addressed filter. Not by the confidence.
- `"scroll to the"` (a fragment) → `scroll_to_element` at **1.00**, past
  scroll's 0.88 speculation bar. Stopped only because "the" grounds to nothing.

Rules 2, 4 and the command filter are load-bearing; the per-action thresholds
are mostly not, at least on this evidence. Do not treat a Jev choice
probability as a measure of whether the answer is right.

## Asking a second question instead of trusting the first

`choice` is decent at *which* action (83–88%). It is the *how sure* that is
useless. So stop asking one question to do two jobs and add a `score` question
— the type built to produce gradation — about the same utterance:

> How clearly does this utterance name one single action from the list, as an
> instruction to a computer?

over five ordered rungs, from "not an instruction at all" to "unmistakable".
It rides along in the request already being sent, so it costs input tokens and
no round trip. Set `VoicePipeline(clarity_floor=2.5)` to turn it on; it is off
by default.

Measured on the same 24 cases, one request per utterance, restricted to the 18
answers that picked an action — blocking a `no_action` answer is free, since
the pipeline rejects those anyway, and counting them makes any bar look far
worse than it is:

```
                 correct (14)   wrong (4)
clarity  mean        3.47          1.74     <- separates
choice p mean        0.99          0.78     <- separates the harmless ones only

 0.23  p=0.52  WRONG  where is the cancel button
 1.58  p=1.00  WRONG  she said we should delete the whole thing
 2.11  p=0.92  ok     open the        | fragments: held anyway on a partial
 2.13  p=1.00  ok     click the sa    |
 2.14  p=1.00  WRONG  scroll to the
 3.02  p=0.60  WRONG  show me the delete button      <- escapes the floor
 3.15 … 3.94          every complete, correct command
```

**A floor at 2.5 blocks three of the four misses, and costs nothing but
fragments** — the two correct answers below it are unfinished transcripts the
gate holds anyway. Both dangerous misses are among the three: `delete_element`
at p=1.00 and a speculative `scroll_to_element` at p=1.00, neither of which any
threshold on the probability could have touched.

The one that escapes is the one that does not matter: "show me the delete
button" resolved to `highlight_element` instead of `scroll_to_element`. Both
are read-only and idempotent, on the same target. Note the pattern in the p
column — the probability was low (0.52, 0.60) exactly on the two harmless
misses and 1.00 on the two dangerous ones. As a gate that is worse than
useless.

Raising the floor to 3.5 catches the fourth miss and starts eating real
commands (`type my email address in there` at 3.15), which is a bad trade for
an action that was harmless to begin with.

Caveats worth more than the numbers: 18 actionable answers, four of them wrong,
one sample. A bar tuned to four misses is a bar tuned to noise. The rung
wording is load-bearing. The two runs also disagreed on one case — "where is
the cancel button" was right in the batched run and wrong here — so some of
this is run-to-run variance. Treat 2.5 as a starting point and run
`scripts/probe_clarity.py` against your own set before shipping it.

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
