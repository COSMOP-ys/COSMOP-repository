"""DOM grounding: spoken description -> on-screen candidates, without pixels.

In a browser the accessibility tree already says what every control is called,
what kind of thing it is, where it sits and what colour it is. Reaching for a
segmentation model there is paying $2.50 per thousand screenshots, and a round
trip, for information the page is willing to hand over for free - and it ships a
picture of the user's screen to a third party to get it.

So DOM is the default grounder and SAM 3.1 is the fallback, for the cases the
DOM genuinely cannot answer: <canvas>, video, native windows, or a screenshot of
something that is not this page.

The division of labour is the same either way. This module is a *recall* device:
it shortlists everything the utterance could plausibly mean, cheaply and
locally, and Jev makes the one-of-N pick. Being too eager here is nearly free,
because the closed candidate set is what stops a coordinate being invented.
Being too strict is not recoverable - a target that never makes the shortlist
cannot be chosen.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .grounding import Frame
from .schema import Candidate

MAX_CANDIDATES = 16

# Words that describe the act rather than the thing. Dropping them keeps
# "click the submit button" from matching every button via the word "click".
_VERBS = frozenset(
    """
    click press tap hit push select choose pick open close scroll go move jump
    show hide focus highlight point delete remove clear type enter write fill
    set put take find look see get please now just then let lets want need
    """.split()
)
_STOPWORDS = frozenset(
    """
    the a an this that these those it its there here to on in into at of for with
    and or but my your our their his her me you i we they is are was were be
    been being do does did done can could would should will shall may might
    up down left right next previous one thing item element
    """.split()
)
# Japanese does not put spaces between words, so a whitespace or [a-z0-9]
# tokenizer returns nothing at all for it - not a poor match, zero candidates.
# Splitting on script boundaries instead gets most of the way there without a
# morphological analyser, because kanji/kana alternation marks word edges:
# 変更を保存ボタンを押して -> 変更 | を | 保存 | ボタン | を | 押 | して.
# The grammatical hiragana runs then drop out as stopwords and what is left is
# the content. Bigrams cover the rest: メール has to match メールアドレス, and
# no run-level comparison does that.
_JA_STOPWORDS = frozenset(
    """
    を は が に へ で と も の や から まで より ね よ か な こと それ これ あれ
    して する した します ください ちょうだい おねがい お願い ある いる です ます
    この その あの どの ここ そこ どこ もの ほう ため
    """.split()
)
# Operation verbs, the same trade the English list makes: a word like 削除 is
# both "the act of deleting" and half the label on the delete button. Dropping
# it costs a little recall on the label and buys not matching every button
# whenever someone says the verb.
_JA_VERBS = frozenset(
    """
    押し 押す 押して クリック タップ 選択 選ん 開い 開く 閉じ 閉じる スクロール
    移動 表示 見せ 教え 削除 消し 消す 入力 打ち 打つ 書い 書く 記入 保存
    ハイライト 強調 探し 探す 見つけ 支払 購入
    """.split()
)

# Spoken words mapped into the vocabulary the snapshot emits, which is English
# because that is what the DOM and getComputedStyle give us. Without this,
# "赤いボタン" cannot reach a node whose colour attribute says "red".
_SYNONYMS = {
    "赤": "red", "あか": "red", "レッド": "red",
    "青": "blue", "あお": "blue", "ブルー": "blue",
    "緑": "green", "みどり": "green", "グリーン": "green",
    "黄": "yellow", "黄色": "yellow", "イエロー": "yellow",
    "白": "white", "しろ": "white", "ホワイト": "white",
    "黒": "black", "くろ": "black", "ブラック": "black",
    "灰": "grey", "灰色": "grey", "グレー": "grey",
    "紫": "purple", "パープル": "purple",
    "橙": "orange", "オレンジ": "orange",
    "桃": "pink", "ピンク": "pink",
    "水色": "teal",
    "ボタン": "button",
    "リンク": "link",
    "欄": "textbox", "入力欄": "textbox", "フィールド": "textbox",
    "チェックボックス": "checkbox",
    "画像": "image",
    "メニュー": "menu",
}

_IGNORED = _VERBS | _STOPWORDS | _JA_STOPWORDS | _JA_VERBS

_HAN = r"一-鿿㐀-䶿"
_HIRAGANA = r"぀-ゟ"
_KATAKANA = r"゠-ヿ"
# One run per script. The boundaries between them are the segmentation.
_TOKEN = re.compile(
    f"[{_HAN}]+|[{_KATAKANA}]+|[{_HIRAGANA}]+|[a-z0-9]+"
)
_CJK_RUN = re.compile(f"\\A[{_HAN}{_KATAKANA}{_HIRAGANA}]+\\Z")
_HIRAGANA_RUN = re.compile(f"\\A[{_HIRAGANA}]+\\Z")


def _tokenize(text: str) -> list[str]:
    """Every surface form worth matching on, in order, with repeats kept.

    NFKC first so that full-width ＡＢＣ and half-width ｶﾀｶﾅ reach the same
    tokens their normal forms would.
    """
    out: list[str] = []
    for run in _TOKEN.findall(unicodedata.normalize("NFKC", text).lower()):
        out.append(run)
        if _CJK_RUN.match(run) and len(run) > 2:
            # メールアドレス has to be reachable from メール, and a whole-run
            # comparison never gets there.
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


@dataclass(frozen=True)
class DomNode:
    """One element from a page snapshot.

    `ref` is written back onto the element as `data-jev-ref`, so the executor
    can find exactly the node that was grounded rather than re-running a
    description match against a page that may have moved on.
    """

    ref: str
    role: str = ""
    name: str = ""
    text: str = ""
    box: tuple[float, float, float, float] | None = None
    visible: bool = True
    # Rendered but scrolled out of view. Still a legitimate target - it is the
    # whole point of "scroll to ..." - so it is described, not filtered.
    in_viewport: bool = True
    enabled: bool = True
    # Free-form descriptors worth matching on: colour, placeholder, title, state.
    attrs: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DomNode":
        return cls(
            ref=str(raw["ref"]),
            role=str(raw.get("role") or ""),
            name=str(raw.get("name") or ""),
            text=str(raw.get("text") or ""),
            box=_box(raw.get("box")),
            visible=bool(raw.get("visible", True)),
            in_viewport=bool(raw.get("inViewport", True)),
            enabled=bool(raw.get("enabled", True)),
            attrs={str(k): str(v) for k, v in (raw.get("attrs") or {}).items() if v},
        )

    def describe(self) -> str:
        """The sentence Jev is shown for this option."""
        bits = [self.role or "element"]
        if self.name:
            bits.append(f'"{self.name}"')
        extra = ", ".join(f"{k}: {v}" for k, v in sorted(self.attrs.items()))
        if extra:
            bits.append(f"({extra})")
        if self.text and self.text != self.name:
            bits.append(f"- {self.text[:80]}")
        if not self.in_viewport:
            # Said in words rather than put in `attrs`, so it describes the
            # option to Jev without adding "screen" to the lexical haystack.
            bits.append("[scrolled out of view]")
        return " ".join(bits)

    def haystack(self) -> set[str]:
        parts = [self.name, self.text, self.role, *self.attrs.values(), *self.attrs.keys()]
        return {t for part in parts for t in _tokenize(part)}


def _box(raw: Any) -> tuple[float, float, float, float] | None:
    """A rectangle, or nothing. This parses whatever a page chose to send.

    The position is a hint for the description Jev reads, never how the action
    finds its element, so a missing or nonsensical box costs a little context
    and must not take the whole snapshot down with it.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    return tuple(values) if all(math.isfinite(v) for v in values) else None


def content_tokens(prompt: str) -> list[str]:
    """What the speaker said, minus the words that describe the act of saying it."""
    seen: list[str] = []
    for token in _tokenize(prompt):
        token = _SYNONYMS.get(token, token)
        if _too_short(token) or token in _IGNORED or token in seen:
            continue
        seen.append(token)
    return seen


def _too_short(token: str) -> bool:
    """One Latin letter says nothing; one kanji often says everything.

    A run of hiragana that short is grammar - a particle or an inflection - so
    it goes whichever way the length rule sends it.
    """
    if _HIRAGANA_RUN.match(token):
        return len(token) < 3
    if _CJK_RUN.match(token):
        return False
    return len(token) < 2


def _weights(nodes: Sequence[DomNode], tokens: Sequence[str]) -> dict[str, float]:
    """How much each spoken word narrows things down, on this page.

    "button" on a page of buttons says almost nothing; "bicycle" says almost
    everything. Plain inverse document frequency over the candidate pool, which
    is enough to stop a generic noun from scoring like a name.
    """
    total = len(nodes) or 1
    hays = [n.haystack() for n in nodes]
    weights: dict[str, float] = {}
    for token in tokens:
        df = sum(1 for hay in hays if token in hay)
        weights[token] = math.log(1 + total / df) if df else 0.0
    return weights


def _score(
    node: DomNode,
    tokens: Sequence[str],
    prompt_lower: str,
    weights: Mapping[str, float],
    pool_size: int,
) -> float:
    """How much of what they said this node accounts for, as a rough confidence.

    Not a calibrated probability, and deliberately never 1.0. Two things hold it
    down: how much of the utterance's *informative* weight the node matched, and
    how many words that took. One overlapping word is evidence, not proof, so it
    lands low enough that `combine()` pushes the plan to CONFIRM instead of
    running it - which is the right answer when a single common noun is all
    that lines up.
    """
    if not tokens:
        return 0.0
    hay = node.haystack()
    matched = [t for t in tokens if t in hay]
    if not matched:
        return 0.0

    # Said by name, and the name is most of what was said: no ambiguity to weigh.
    name = node.name.strip().lower()
    if len(name) >= 3 and name in prompt_lower:
        name_tokens = {_SYNONYMS.get(t, t) for t in _tokenize(name)}
        if len(name_tokens & set(tokens)) / len(tokens) >= 0.6:
            return 0.97

    total_weight = sum(weights.values())
    matched_weight = sum(weights[t] for t in matched)
    coverage = matched_weight / total_weight if total_weight else len(matched) / len(tokens)
    # How discriminating that evidence was, against the most a single word can
    # be worth here: a token belonging to exactly one node maxes this out, a
    # word every button shares barely moves it. Word *count* is the wrong
    # measure - "the email field" identifies one input as surely as a full name.
    peak = math.log(1 + pool_size) if pool_size else 0.0
    strength = min(1.0, matched_weight / peak) if peak else 0.0
    return round(min(0.95, 0.12 + 0.83 * coverage * strength), 4)


class DomGrounder:
    """Shortlists accessibility-tree nodes that match the utterance.

    `provider` is called per grounding and returns the current snapshot, so the
    candidates are the page as it is now, not as it was when the user started
    talking. Pass `Frame`-free: nothing here needs a screenshot.
    """

    needs_frame = False

    def __init__(
        self,
        provider: Callable[[], Sequence[DomNode | Mapping[str, Any]]],
        *,
        max_candidates: int = MAX_CANDIDATES,
        include_hidden: bool = False,
    ) -> None:
        self._provider = provider
        self._max_candidates = max_candidates
        self._include_hidden = include_hidden
        self.last_snapshot: list[DomNode] = []

    def ground(self, prompt: str, frame: Frame | None = None) -> Sequence[Candidate]:
        nodes = [
            n if isinstance(n, DomNode) else DomNode.from_dict(n) for n in self._provider()
        ]
        self.last_snapshot = nodes
        tokens = content_tokens(prompt)
        prompt_lower = prompt.lower()

        pool = [n for n in nodes if self._include_hidden or n.visible]
        weights = _weights(pool, tokens)
        scored: list[tuple[float, DomNode]] = []
        for node in pool:
            score = _score(node, tokens, prompt_lower, weights, len(pool))
            if score > 0.0:
                scored.append((score, node))

        # Stable order: best first, then document order for equal scores, so the
        # same page and the same words always produce the same candidate list.
        order = {id(n): i for i, n in enumerate(nodes)}
        scored.sort(key=lambda pair: (-pair[0], order[id(pair[1])]))
        return [
            Candidate(ref=node.ref, label=node.describe(), box=node.box, score=score)
            for score, node in scored[: self._max_candidates]
        ]


# Page-side extractor. Evaluates to an array of node objects matching
# `DomNode.from_dict`, and stamps `data-jev-ref` on each element so the same
# node can be acted on afterwards. Deliberately plain ES5-ish so it can be
# dropped into any evaluate()/executeScript() without a build step.
DOM_SNAPSHOT_JS = r"""
(() => {
  // Scope the snapshot to a subtree by setting window.__jevRoot. A console that
  // hosts both the app and its own controls needs this, or its microphone
  // button becomes a candidate for "click the button".
  const ROOT = window.__jevRoot || document;
  const INTERACTIVE = 'a,button,input,select,textarea,summary,[role],[onclick],[tabindex],[contenteditable=""],[contenteditable="true"]';
  const ROLE_BY_TAG = {
    A: 'link', BUTTON: 'button', INPUT: 'textbox', SELECT: 'combobox',
    TEXTAREA: 'textbox', SUMMARY: 'disclosure', IMG: 'image', LABEL: 'label',
  };
  const INPUT_ROLE = {
    checkbox: 'checkbox', radio: 'radio', submit: 'button', button: 'button',
    reset: 'button', range: 'slider', file: 'file input', search: 'searchbox',
  };

  const colourName = (css) => {
    const m = /rgba?\(([^)]+)\)/.exec(css || '');
    if (!m) return '';
    const p = m[1].split(',').map(Number);
    if (p.length > 3 && p[3] < 0.1) return '';
    const [r, g, b] = p.map((v) => v / 255);
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    const l = (max + min) / 2;
    if (d < 0.12) return l > 0.9 ? 'white' : l < 0.15 ? 'black' : 'grey';
    let h = 0;
    if (max === r) h = 60 * (((g - b) / d) % 6);
    else if (max === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
    if (h < 0) h += 360;
    const names = [[15,'red'],[45,'orange'],[70,'yellow'],[160,'green'],
                   [200,'teal'],[255,'blue'],[290,'purple'],[335,'pink'],[360,'red']];
    for (const [edge, name] of names) if (h < edge) return name;
    return '';
  };

  const accessibleName = (el) => {
    const labelled = el.getAttribute('aria-labelledby');
    if (labelled) {
      const parts = labelled.split(/\s+/)
        .map((id) => document.getElementById(id))
        .filter(Boolean)
        .map((n) => n.innerText || '');
      if (parts.join(' ').trim()) return parts.join(' ').trim();
    }
    return (
      el.getAttribute('aria-label') ||
      el.getAttribute('alt') ||
      el.getAttribute('title') ||
      (el.labels && el.labels[0] && el.labels[0].innerText) ||
      el.getAttribute('placeholder') ||
      (el.value && el.type !== 'password' ? el.value : '') ||
      (el.innerText || '').trim().slice(0, 120)
    ).trim();
  };

  // Refs are reused across snapshots so a ref survives a re-snapshot mid
  // utterance, but only when they look like ours: the attribute is in the
  // page's reach, and a ref is about to be pasted into a selector. If a page
  // ever does duplicate one, Playwright's strict locators fail on the two
  // matches rather than picking the decoy.
  const OURS = /^n\d+$/;
  const taken = new Set(
    Array.from(ROOT.querySelectorAll('[data-jev-ref]'))
      .map((e) => e.getAttribute('data-jev-ref'))
      .filter((r) => OURS.test(r || ''))
  );

  const vw = innerWidth || document.documentElement.clientWidth || 0;
  const vh = innerHeight || document.documentElement.clientHeight || 0;
  const sized = vw > 0 && vh > 0;

  const out = [];
  let n = 0;
  for (const el of ROOT.querySelectorAll(INTERACTIVE)) {
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const visible =
      rect.width > 1 && rect.height > 1 &&
      style.visibility !== 'hidden' && style.display !== 'none' &&
      parseFloat(style.opacity || '1') > 0.05;
    if (!visible) continue;
    // A hidden or zero-sized window reports innerWidth 0, and dividing by it
    // yields Infinity, which JSON.stringify turns into null - so the box
    // arrives as [null,null,null,null] and the reader has to cope. Better not
    // to send it: an unknown position is not the same as position zero.
    const inViewport = sized
      ? rect.bottom > 0 && rect.right > 0 && rect.top < vh && rect.left < vw
      : true;

    let ref = el.getAttribute('data-jev-ref');
    if (!ref || !OURS.test(ref)) {
      do { ref = 'n' + n++; } while (taken.has(ref));
      taken.add(ref);
      el.setAttribute('data-jev-ref', ref);
    }

    const attrs = {};
    const bg = colourName(style.backgroundColor);
    const fg = colourName(style.color);
    if (bg) attrs.colour = bg;
    if (fg && fg !== bg) attrs['text colour'] = fg;
    if (el.getAttribute('placeholder')) attrs.placeholder = el.getAttribute('placeholder');
    if (el.checked) attrs.state = 'checked';
    if (el.getAttribute('aria-expanded')) attrs.expanded = el.getAttribute('aria-expanded');

    out.push({
      ref,
      role: el.getAttribute('role') ||
            (el.tagName === 'INPUT' ? (INPUT_ROLE[el.type] || 'textbox') : '') ||
            ROLE_BY_TAG[el.tagName] || el.tagName.toLowerCase(),
      name: accessibleName(el),
      text: (el.innerText || '').trim().slice(0, 160),
      box: sized
        ? [rect.left / vw, rect.top / vh, rect.right / vw, rect.bottom / vh]
        : null,
      visible: true,
      inViewport,
      enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
      attrs,
    });
  }
  return out;
})()
"""
