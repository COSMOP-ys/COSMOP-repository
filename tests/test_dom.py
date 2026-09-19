import unittest

from jev_voice_cv.dom import DomGrounder, DomNode, content_tokens
from jev_voice_cv.jev import MockJev
from jev_voice_cv.pipeline import VoicePipeline
from jev_voice_cv.schema import ActionSpec, Decision, PlanKind

NODES = [
    DomNode("n0", role="button", name="Delete account", attrs={"colour": "red"}),
    DomNode("n1", role="button", name="Save changes", attrs={"colour": "green"}),
    DomNode("n2", role="button", name="Cancel", attrs={"colour": "grey"}),
    DomNode("n3", role="textbox", name="Email", attrs={"placeholder": "you@example.com"}),
    DomNode("n4", role="link", name="Read the documentation"),
    DomNode("n5", role="button", name="Delete everything", visible=False,
            attrs={"colour": "red"}),
]

CLICK = ActionSpec("click_element", "press a control", execute_threshold=0.85,
                   speculate_threshold=0.97, needs_target=True)


def grounder(**kwargs) -> DomGrounder:
    return DomGrounder(lambda: NODES, **kwargs)


class TokenTest(unittest.TestCase):
    def test_the_verb_is_not_part_of_the_description(self):
        self.assertEqual(content_tokens("click the save changes button"), ["save", "changes", "button"])

    def test_repeats_collapse_and_short_noise_drops(self):
        self.assertEqual(content_tokens("the red red x button"), ["red", "button"])


class DomGroundingTest(unittest.TestCase):
    def test_colour_from_computed_style_is_matchable(self):
        refs = [c.ref for c in grounder().ground("click the red button")]
        self.assertEqual(refs[:2], ["n0", "n1"])
        self.assertIn("colour: red", grounder().ground("click the red button")[0].label)

    def test_hidden_nodes_are_not_candidates(self):
        self.assertNotIn("n5", [c.ref for c in grounder().ground("delete everything")])
        self.assertIn("n5", [c.ref for c in grounder(include_hidden=True).ground("delete everything")])

    def test_a_name_said_in_full_outscores_a_word_every_button_shares(self):
        [named] = [c for c in grounder().ground("click save changes") if c.ref == "n1"]
        [generic] = [c for c in grounder().ground("click the button") if c.ref == "n1"]
        self.assertGreater(named.score, 0.9)
        self.assertLess(generic.score, 0.7)

    def test_one_discriminating_word_is_enough_to_score_high(self):
        # "email" belongs to exactly one node on this page, so it identifies it
        # as surely as a full name would. Word count is not the measure.
        [only] = grounder().ground("scroll to the email field")
        self.assertEqual(only.ref, "n3")
        self.assertGreater(only.score, 0.9)

    def test_a_shared_word_alone_leaves_room_for_the_gate_to_ask(self):
        for candidate in grounder().ground("click the button"):
            self.assertLess(candidate.score, 0.7)

    def test_no_match_is_ever_certain(self):
        for prompt in ("click save changes", "click the red delete account button"):
            for candidate in grounder().ground(prompt):
                self.assertLess(candidate.score, 1.0)

    def test_nothing_matches_rather_than_everything(self):
        self.assertEqual(grounder().ground("the bicycle in the photo"), [])

    def test_the_shortlist_is_capped(self):
        self.assertEqual(len(grounder(max_candidates=2).ground("button")), 2)

    def test_equal_scores_keep_document_order(self):
        refs = [c.ref for c in grounder().ground("button")]
        self.assertEqual(refs, ["n0", "n1", "n2"])

    def test_dicts_from_the_page_are_accepted_directly(self):
        raw = [{"ref": "x", "role": "button", "name": "Ok", "box": [0, 0, 1, 1]}]
        [candidate] = DomGrounder(lambda: raw).ground("click ok")
        self.assertEqual(candidate.box, (0.0, 0.0, 1.0, 1.0))


class DomPipelineTest(unittest.TestCase):
    """The DOM grounder needs no screenshot, so no frame is captured at all."""

    def _pipeline(self, target_p: float) -> VoicePipeline:
        def handler(question, state, options):
            if list(options) == ["yes", "no"]:
                return Decision("yes", {"yes": 0.99, "no": 0.01})
            if "candidate" in question.lower():
                return Decision(options[0], {o: target_p if i == 0 else (1 - target_p)
                                             for i, o in enumerate(options)})
            return Decision("click_element", {"click_element": 0.99, "no_action": 0.01})

        return VoicePipeline(actions=[CLICK], jev=MockJev(handler), grounder=grounder())

    def test_a_frameless_ticket_still_grounds(self):
        pipeline = self._pipeline(target_p=0.99)
        plan = pipeline.resolve(pipeline.submit("click the red button", final=True))
        self.assertIs(plan.kind, PlanKind.EXECUTE)
        self.assertEqual(plan.target.ref, "n0")

    def test_an_ambiguous_red_button_is_confirmed_not_clicked(self):
        pipeline = self._pipeline(target_p=0.80)
        plan = pipeline.resolve(pipeline.submit("click the red button", final=True))
        self.assertIs(plan.kind, PlanKind.CONFIRM)


JA_NODES = [
    DomNode("j0", role="button", name="アカウントを削除",
            attrs={"colour": "red"}),
    DomNode("j1", role="button", name="変更を保存",
            attrs={"colour": "green"}),
    DomNode("j2", role="button", name="キャンセル",
            attrs={"colour": "grey"}),
    DomNode("j3", role="textbox", name="メールアドレス",
            attrs={"placeholder": "you@example.com"}),
    DomNode("j4", role="link", name="ドキュメントを読む"),
]


def ja_grounder(**kwargs) -> DomGrounder:
    return DomGrounder(lambda: JA_NODES, **kwargs)


class JapaneseTokenTest(unittest.TestCase):
    """Japanese writes no spaces, so the script boundaries are the segmentation."""

    def test_a_sentence_splits_at_the_kanji_kana_boundaries(self):
        # 変更を保存ボタンを押して
        tokens = content_tokens("変更を保存ボタンを押して")
        self.assertIn("変更", tokens)          # the content survives
        self.assertNotIn("を", tokens)         # the particle does not
        self.assertNotIn("して", tokens)       # nor the inflection

    def test_a_single_kanji_is_not_too_short_to_keep(self):
        # 赤 alone carries the whole description; one Latin letter would not.
        self.assertIn("red", content_tokens("赤いボタン"))

    def test_colour_and_role_words_reach_the_english_the_dom_emits(self):
        tokens = content_tokens("赤いボタンをクリック")
        self.assertIn("red", tokens)
        self.assertIn("button", tokens)

    def test_full_width_and_half_width_normalise(self):
        self.assertEqual(content_tokens("ＳＡＶＥ"), ["save"])
        self.assertIn("キャンセル", content_tokens("ｷｬﾝｾﾙ"))

    def test_bigrams_let_a_prefix_reach_a_longer_name(self):
        # メール has to be able to find メールアドレス.
        self.assertTrue(set(content_tokens("メール")) & JA_NODES[3].haystack())


class JapaneseGroundingTest(unittest.TestCase):
    def test_a_name_said_in_full_wins(self):
        best = ja_grounder().ground("変更を保存ボタンを押して")[0]
        self.assertEqual(best.ref, "j1")
        self.assertGreater(best.score, 0.9)

    def test_colour_alone_picks_the_right_button(self):
        self.assertEqual(ja_grounder().ground("赤いボタンをクリック")[0].ref, "j0")

    def test_a_shortened_field_name_still_grounds(self):
        [only] = ja_grounder().ground("メール欄までスクロールして")
        self.assertEqual(only.ref, "j3")

    def test_a_katakana_name_said_bare_grounds(self):
        self.assertEqual(ja_grounder().ground("キャンセル")[0].ref, "j2")

    def test_nothing_on_the_page_matches_rather_than_everything(self):
        self.assertEqual(ja_grounder().ground("自転車を押して"), [])

    def test_the_english_path_is_unchanged_by_any_of_this(self):
        self.assertEqual([c.ref for c in grounder().ground("click the red button")][:2], ["n0", "n1"])


class MalformedSnapshotTest(unittest.TestCase):
    """The snapshot is whatever a page chose to send, so parse it defensively.

    A hidden or zero-sized window reports innerWidth 0; the extractor divides
    by it, gets Infinity, and JSON.stringify writes null. One node like that
    used to raise and take the entire utterance down with it.
    """

    def test_a_box_of_nulls_costs_the_position_and_nothing_else(self):
        node = DomNode.from_dict({"ref": "n0", "name": "Save", "box": [None, None, None, None]})
        self.assertIsNone(node.box)
        self.assertEqual(node.name, "Save")

    def test_every_shape_a_page_might_send_is_survivable(self):
        for box in ([float("inf")] * 4, [float("nan")] * 4, "nope", [1, 2], {}, None, [1, 2, 3, "x"]):
            self.assertIsNone(DomNode.from_dict({"ref": "n0", "box": box}).box)

    def test_a_positionless_node_is_still_a_candidate(self):
        grounder = DomGrounder(lambda: [{"ref": "n0", "role": "button", "name": "Save changes",
                                         "box": [None, None, None, None]}])
        [only] = grounder.ground("click save changes")
        self.assertEqual(only.ref, "n0")
        self.assertIsNone(only.box)


class LabelWordsAreNotVerbsTest(unittest.TestCase):
    """A word that appears on a control must not be thrown away as a gesture.

    保存 was in the verb stoplist, so "保存ボタンを押して" reduced to nothing but
    `button`, every button tied, and the first one in the document won - which
    was the delete button. idf already handles the case the stoplist was meant
    to cover: a word every control carries scores low by itself.
    """

    def test_a_verb_that_is_also_a_label_still_targets(self):
        [best, *_] = ja_grounder().ground("保存ボタンを押して")
        self.assertEqual(best.ref, "j1")

    def test_the_same_holds_for_delete(self):
        self.assertEqual(ja_grounder().ground("削除して")[0].ref, "j0")

    def test_a_pure_gesture_word_still_contributes_nothing(self):
        # クリック appears on no control here, so it must not decide anything.
        self.assertNotIn("クリック", content_tokens("キャンセルをクリック"))
        self.assertEqual(ja_grounder().ground("キャンセルをクリック")[0].ref, "j2")


if __name__ == "__main__":
    unittest.main()
