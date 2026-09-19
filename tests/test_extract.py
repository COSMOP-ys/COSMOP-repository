import unittest

from jev_voice_cv import extract
from jev_voice_cv.jev import MockJev
from jev_voice_cv.pipeline import VoicePipeline
from jev_voice_cv.schema import ActionSpec, Candidate, Decision, PlanKind

TYPE = ActionSpec("type_text", "enter dictated text into a named field",
                  execute_threshold=0.85, speculate_threshold=0.97,
                  needs_target=True, needs_text_arg=True)

# メールアドレスにテストを入力  -  "enter TEST in the email address field"
JA = "メールアドレスにテストを入力"
TEST = "テスト"


class SpanTest(unittest.TestCase):
    def test_every_answer_is_something_the_speaker_actually_said(self):
        for span in extract.spans("type hello world into the email box"):
            self.assertIn(span, "type hello world into the email box")

    def test_the_dictated_part_is_among_the_options(self):
        self.assertIn("hello world", extract.spans("type hello world into the email box"))

    def test_japanese_splits_without_spaces(self):
        self.assertIn(TEST, extract.spans(JA))

    def test_longest_first_so_the_cap_bites_at_the_short_end(self):
        got = extract.spans("alpha beta gamma")
        self.assertEqual(got[0], "alpha beta gamma")
        self.assertGreater(len(got[0]), len(got[-1]))

    def test_an_empty_utterance_offers_nothing(self):
        self.assertEqual(extract.spans("   ...   "), [])

    def test_the_option_set_stays_inside_jev_s_one_of_n_limit(self):
        long_one = " ".join(f"word{i}" for i in range(60))
        self.assertLessEqual(len(extract.options(long_one)), 255)

    def test_there_is_always_a_way_to_say_it_was_not_said(self):
        self.assertIn(extract.NO_TEXT, extract.options("press the button"))

    def test_the_escape_hatch_resolves_to_nothing(self):
        self.assertIsNone(extract.resolve(extract.NO_TEXT, ["a", "b"]))

    def test_an_answer_outside_the_option_set_is_refused(self):
        # The point of a closed set is that nothing else can come back.
        self.assertIsNone(extract.resolve("something never said", ["a", "b"]))


class PipelineTextTest(unittest.TestCase):
    """The text question rides in the same request as the target question."""

    class _Grounder:
        needs_frame = False

        def ground(self, prompt, frame=None):
            return [Candidate("f1", 'textbox "email"', score=0.95)]

    @staticmethod
    def _jev(span: str | None):
        def handler(question, state, options):
            if list(options) == ["yes", "no"]:
                return Decision("yes", {"yes": 0.99, "no": 0.01})
            if extract.NO_TEXT in options:
                picked = span if span in options else extract.NO_TEXT
                return Decision(picked, {o: 0.9 if o == picked else 0.0 for o in options})
            return Decision("type_text", {"type_text": 0.99, "no_action": 0.01})

        return MockJev(handler)

    def _pipeline(self, span):
        return VoicePipeline(actions=[TYPE], jev=self._jev(span), grounder=self._Grounder())

    def test_the_chosen_span_arrives_on_the_plan(self):
        pipeline = self._pipeline(TEST)
        plan = pipeline.resolve(pipeline.submit(JA, final=True))
        self.assertIs(plan.kind, PlanKind.EXECUTE)
        self.assertEqual(plan.text_arg, TEST)
        self.assertEqual(plan.text_arg_from, "transcript-span")

    def test_it_costs_no_extra_round_trip(self):
        pipeline = self._pipeline(TEST)
        pipeline.resolve(pipeline.submit(JA, final=True))
        # One request for filter+intent, one for text. The target needed no
        # question because the shortlist had a single entry.
        self.assertEqual(pipeline.jev.batches, [["intent", "addressed"], ["text"]])

    def test_an_utterance_with_no_content_asks_instead_of_running(self):
        pipeline = self._pipeline(None)
        plan = pipeline.resolve(pipeline.submit(JA, final=True))
        self.assertIs(plan.kind, PlanKind.CONFIRM)
        self.assertIsNone(plan.text_arg)
        self.assertEqual(plan.reason, "nothing in the utterance is the text to enter")

    def test_mid_utterance_it_waits_rather_than_asking(self):
        pipeline = self._pipeline(None)
        self.assertIs(pipeline.resolve(pipeline.submit(JA, final=False)).kind, PlanKind.HOLD)

    def test_extraction_can_be_turned_off(self):
        pipeline = VoicePipeline(actions=[TYPE], jev=self._jev(TEST),
                                 grounder=self._Grounder(), text_extraction=False)
        plan = pipeline.resolve(pipeline.submit(JA, final=True))
        self.assertIsNone(plan.text_arg)
        self.assertEqual(plan.text_arg_from, "llm")
        self.assertEqual(pipeline.jev.batches, [["intent", "addressed"]])


if __name__ == "__main__":
    unittest.main()
