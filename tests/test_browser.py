"""End-to-end against a real browser: snapshot -> ground -> gate -> act.

Skipped unless Playwright and a browser are installed, so the default suite
stays dependency-free. Run it with:

    .venv/Scripts/python -m unittest tests.test_browser
"""

from __future__ import annotations

import pathlib
import unittest
from dataclasses import replace
from typing import Any, Mapping, Sequence

from jev_voice_cv import extract
from jev_voice_cv.dom import DomGrounder
from jev_voice_cv.jev import MockJev
from jev_voice_cv.pipeline import VoicePipeline
from jev_voice_cv.schema import ActionSpec, Decision, PlanKind

try:  # pragma: no cover - environment probe
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

from jev_voice_cv.playwright_exec import (  # noqa: E402 - after the probe on purpose
    ExecutionError,
    PlaywrightExecutor,
    dom_provider,
    selector_for,
    snapshot,
)

FIXTURE = (pathlib.Path(__file__).parent / "fixtures" / "page.html").resolve().as_uri()

CLICK = ActionSpec("click_element", "press a control on the page",
                   execute_threshold=0.85, speculate_threshold=0.97, needs_target=True)
SCROLL = ActionSpec("scroll_to_element", "bring something into view", read_only=True,
                    idempotent=True, execute_threshold=0.70, speculate_threshold=0.90,
                    needs_target=True)
TYPE = ActionSpec("type_text", "enter dictated text", execute_threshold=0.85,
                  speculate_threshold=0.97, needs_target=True, needs_text_arg=True)


def _launch(playwright):
    """Prefer a browser the machine already has over a downloaded one."""
    for kwargs in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return playwright.chromium.launch(**kwargs)
        except Exception:
            continue
    return None


def decide(action: str, *, intent_p: float = 0.99, target_p: float = 0.99,
           span: str | None = None):
    def handler(question: str, state: Mapping[str, Any], options: Sequence[str]) -> Decision:
        if list(options) == ["yes", "no"]:
            return Decision("yes", {"yes": 0.99, "no": 0.01})
        if extract.NO_TEXT in options:
            picked = span if span in options else extract.NO_TEXT
            return Decision(picked, {o: 0.9 if o == picked else 0.0 for o in options})
        if "candidate" in question.lower():
            rest = (1 - target_p) / max(len(options) - 1, 1)
            return Decision(options[0], {o: target_p if i == 0 else rest
                                         for i, o in enumerate(options)})
        rest = (1 - intent_p) / max(len(options) - 1, 1)
        return Decision(action, {o: intent_p if o == action else rest for o in options})

    return handler


@unittest.skipIf(sync_playwright is None, "playwright is not installed")
class BrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pw = sync_playwright().start()
        cls.browser = _launch(cls._pw)
        if cls.browser is None:
            cls._pw.stop()
            raise unittest.SkipTest("no chromium-family browser available")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls._pw.stop()

    def setUp(self):
        self.page = self.browser.new_page()
        self.page.goto(FIXTURE)

    def tearDown(self):
        self.page.close()

    def pipeline(self, action: ActionSpec, handler) -> VoicePipeline:
        return VoicePipeline(
            actions=[action],
            jev=MockJev(handler),
            grounder=DomGrounder(dom_provider(self.page)),
        )

    # -- snapshot ------------------------------------------------------------

    def test_snapshot_reads_names_colours_and_roles_off_the_page(self):
        by_name = {n.name: n for n in snapshot(self.page)}
        self.assertEqual(by_name["Delete account"].attrs["colour"], "red")
        self.assertEqual(by_name["Save changes"].attrs["colour"], "green")
        self.assertEqual(by_name["Delete account"].role, "button")
        # aria-label wins over the visible text.
        self.assertIn("Submit the order", by_name)
        # The <label> names the field; the placeholder stays as a descriptor.
        self.assertEqual(by_name["Email"].role, "textbox")
        self.assertEqual(by_name["Email"].attrs["placeholder"], "you@example.com")

    def test_display_none_elements_are_not_in_the_snapshot(self):
        self.assertNotIn("Delete everything", {n.name for n in snapshot(self.page)})

    def test_an_element_below_the_fold_is_a_candidate_and_says_so(self):
        # Filtering to the viewport would delete the only target that could
        # ever satisfy "scroll to ...".
        submit = {n.name: n for n in snapshot(self.page)}["Submit the order"]
        self.assertTrue(submit.visible)
        self.assertFalse(submit.in_viewport)
        self.assertIn("scrolled out of view", submit.describe())

    def test_refs_survive_a_second_snapshot(self):
        first = {n.name: n.ref for n in snapshot(self.page)}
        second = {n.name: n.ref for n in snapshot(self.page)}
        self.assertEqual(first, second)

    # -- grounding -----------------------------------------------------------

    def test_colour_and_name_pick_the_right_button_out_of_four(self):
        grounder = DomGrounder(dom_provider(self.page))
        best = grounder.ground("click the red delete account button")[0]
        node = {n.ref: n for n in grounder.last_snapshot}[best.ref]
        self.assertEqual(node.name, "Delete account")

    # -- execution -----------------------------------------------------------

    def test_a_gated_click_actually_presses_the_button(self):
        pipeline = self.pipeline(CLICK, decide("click_element"))
        plan = pipeline.resolve(pipeline.submit("click save changes", final=True))
        self.assertIs(plan.kind, PlanKind.EXECUTE)
        PlaywrightExecutor(self.page).run(plan)
        self.assertEqual(self.page.inner_text("#log"), "save;")

    def test_scroll_moves_the_viewport(self):
        pipeline = self.pipeline(SCROLL, decide("scroll_to_element"))
        plan = pipeline.resolve(pipeline.submit("scroll to submit the order", final=True))
        self.assertIs(plan.kind, PlanKind.EXECUTE)
        PlaywrightExecutor(self.page).run(plan)
        self.assertGreater(self.page.evaluate("window.scrollY"), 500)

    def test_the_dictated_words_land_in_the_grounded_field(self):
        # No with_text(): the span comes out of the transcript itself.
        pipeline = self.pipeline(TYPE, decide("type_text", span="hello world"))
        plan = pipeline.resolve(
            pipeline.submit("type hello world into the email box", final=True)
        )
        self.assertIs(plan.kind, PlanKind.EXECUTE)
        self.assertEqual(plan.text_arg, "hello world")
        PlaywrightExecutor(self.page).run(plan)
        self.assertEqual(self.page.input_value("#email"), "hello world")

    def test_a_caller_can_still_supply_the_text_itself(self):
        pipeline = self.pipeline(TYPE, decide("type_text", span="hello world"))
        plan = pipeline.resolve(
            pipeline.submit("type hello world into the email box", final=True)
        )
        PlaywrightExecutor(self.page).run(plan.with_text("someone@example.com"))
        self.assertEqual(self.page.input_value("#email"), "someone@example.com")

    def test_an_utterance_with_nothing_dictated_asks_rather_than_running(self):
        pipeline = self.pipeline(TYPE, decide("type_text"))
        plan = pipeline.resolve(pipeline.submit("type into the email box", final=True))
        self.assertIs(plan.kind, PlanKind.CONFIRM)
        self.assertEqual(plan.reason, "nothing in the utterance is the text to enter")

    def test_a_text_action_reaching_the_executor_without_text_still_refuses(self):
        pipeline = self.pipeline(TYPE, decide("type_text", span="hello world"))
        plan = pipeline.resolve(
            pipeline.submit("type hello world into the email box", final=True)
        )
        with self.assertRaises(ExecutionError):
            PlaywrightExecutor(self.page).run(replace(plan, text_arg=None))

    def test_a_target_that_left_the_page_fails_instead_of_hitting_its_replacement(self):
        pipeline = self.pipeline(CLICK, decide("click_element"))
        plan = pipeline.resolve(pipeline.submit("click save changes", final=True))
        self.page.evaluate("document.getElementById('save').remove()")
        with self.assertRaises(ExecutionError):
            PlaywrightExecutor(self.page).run(plan)
        self.assertEqual(self.page.inner_text("#log"), "")

    # -- Japanese ------------------------------------------------------------

    def test_japanese_labels_come_back_whole_from_the_page(self):
        by_name = {n.name: n for n in snapshot(self.page)}
        delete = by_name["アカウントを削除"]
        self.assertEqual(delete.role, "button")
        self.assertEqual(delete.attrs["colour"], "red")
        # The <label> names the field in Japanese too.
        self.assertEqual(by_name["メールアドレス"].role, "textbox")

    def test_a_japanese_command_grounds_and_clicks_the_right_button(self):
        pipeline = self.pipeline(CLICK, decide("click_element"))
        # 変更を保存ボタンを押して - "press the save changes button"
        plan = pipeline.resolve(
            pipeline.submit("変更を保存ボタンを押して", final=True)
        )
        self.assertIs(plan.kind, PlanKind.EXECUTE)
        PlaywrightExecutor(self.page).run(plan)
        self.assertEqual(self.page.inner_text("#log"), "ja-save;")

    def test_a_japanese_colour_reaches_the_english_computed_style(self):
        pipeline = self.pipeline(CLICK, decide("click_element"))
        # 赤いボタンを押して - "press the red button". Two are red; the
        # shortlist has to contain both and Jev picks.
        grounder = DomGrounder(dom_provider(self.page))
        refs = [c.ref for c in grounder.ground("赤いボタンを押して")]
        names = {n.ref: n.name for n in grounder.last_snapshot}
        self.assertIn("Delete account", [names[r] for r in refs[:2]])
        self.assertIn("アカウントを削除", [names[r] for r in refs[:2]])
        del pipeline


    def test_an_unregistered_action_is_a_loud_failure(self):
        pipeline = self.pipeline(CLICK, decide("click_element"))
        plan = pipeline.resolve(pipeline.submit("click save changes", final=True))
        executor = PlaywrightExecutor(self.page, handlers={})
        with self.assertRaises(ExecutionError):
            executor.run(plan)

    def test_highlight_outlines_the_element_and_restores_it(self):
        pipeline = self.pipeline(
            ActionSpec("highlight_element", "point something out", read_only=True,
                       idempotent=True, execute_threshold=0.70, speculate_threshold=0.90,
                       needs_target=True),
            decide("highlight_element"),
        )
        plan = pipeline.resolve(pipeline.submit("highlight save changes", final=True))
        PlaywrightExecutor(self.page).run(plan)
        outline = self.page.eval_on_selector("#save", "el => el.style.outline")
        self.assertIn("3px", outline)
        self.page.wait_for_function(
            "() => document.getElementById('save').style.outline === ''", timeout=4000
        )


class SelectorTest(unittest.TestCase):
    def test_a_ref_the_page_could_have_forged_never_reaches_a_selector(self):
        self.assertEqual(selector_for("n12"), '[data-jev-ref="n12"]')
        for hostile in ('n1"], [id="delete', "n1 ", "", "n" * 65):
            with self.assertRaises(ExecutionError):
                selector_for(hostile)


if __name__ == "__main__":
    unittest.main()
