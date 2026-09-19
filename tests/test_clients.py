import unittest
from typing import Any, Mapping, Sequence

from jev_voice_cv.grounding import Frame, GroundingError, Sam31Grounder, _parse_candidates
from jev_voice_cv.jev import (
    GatewayJev,
    JevError,
    MockJev,
    Question,
    _parse_answer,
    _parse_response,
    boolean,
    choice,
    score,
)
from jev_voice_cv.schema import Candidate, Decision


class QuestionTest(unittest.TestCase):
    def test_choice_options_become_the_answer_space(self):
        q = choice("Which action?", ["open", "close"])
        self.assertEqual(
            q.to_wire(),
            {
                "type": "choice",
                "instructions": "Which action?",
                "criteria": {"open": "open", "close": "close"},
            },
        )

    def test_choice_keeps_descriptions_when_given_them(self):
        q = choice("Which action?", {"open": "launch an app"})
        self.assertEqual(q.to_wire()["criteria"], {"open": "launch an app"})

    def test_boolean_criteria_are_optional(self):
        self.assertNotIn("criteria", boolean("Is it?").to_wire())
        self.assertEqual(
            boolean("Did it pass?", true="exit code 0").to_wire()["criteria"],
            {"true": "exit code 0"},
        )

    def test_score_rungs_stay_ordered(self):
        self.assertEqual(score("How urgent?", ["low", "high"]).to_wire()["criteria"], ["low", "high"])

    def test_malformed_questions_are_refused(self):
        with self.assertRaises(JevError):
            Question("vibes", "how does it feel")
        with self.assertRaises(JevError):
            choice("Which?", [])
        with self.assertRaises(JevError):
            score("How much?", ["only one rung"])
        with self.assertRaises(JevError):
            choice("Which?", [f"option-{i}" for i in range(256)])


class ParseAnswerTest(unittest.TestCase):
    def test_boolean_becomes_a_yes_no_distribution(self):
        decision = _parse_answer({"type": "boolean", "probability": 0.99}, 12.0)
        self.assertEqual(decision.choice, "yes")
        self.assertAlmostEqual(decision.probabilities["no"], 0.01)
        self.assertEqual(decision.latency_ms, 12.0)

    def test_boolean_below_half_reads_as_no(self):
        self.assertEqual(_parse_answer({"type": "boolean", "probability": 0.2}, 0.0).choice, "no")

    def test_choice_keeps_the_reported_winner(self):
        decision = _parse_answer(
            {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "other": 0.1}},
            0.0,
        )
        self.assertEqual(decision.choice, "billing")
        self.assertAlmostEqual(decision.confidence, 0.9)

    def test_rounding_that_drops_the_winner_still_yields_a_confidence(self):
        decision = _parse_answer({"type": "choice", "choice": "a", "probabilities": {"b": 1.0}}, 0.0)
        self.assertEqual(decision.confidence, 0.0)

    def test_score_reports_the_scale_position_and_the_top_rung(self):
        decision = _parse_answer(
            {"type": "score", "score": 2.97, "probabilities": {"2": 0.02, "3": 0.98}}, 0.0
        )
        self.assertEqual(decision.choice, "3")
        self.assertAlmostEqual(decision.score, 2.97)

    def test_unknown_shapes_are_errors_rather_than_guesses(self):
        for payload in ({"type": "freeform", "text": "hi"}, "nope", {"type": "score", "score": 1}):
            with self.assertRaises((JevError, KeyError)):
                _parse_answer(payload, 0.0)


class ParseResponseTest(unittest.TestCase):
    questions = {"a": boolean("Is it?"), "b": choice("Which?", ["x", "y"])}

    def test_every_question_gets_its_answer_back(self):
        answers = _parse_response(
            {
                "answers": {
                    "a": {"type": "boolean", "probability": 0.7},
                    "b": {"type": "choice", "choice": "x", "probabilities": {"x": 0.8, "y": 0.2}},
                }
            },
            self.questions,
            1.0,
        )
        self.assertEqual(sorted(answers), ["a", "b"])

    def test_a_missing_answer_is_an_error(self):
        with self.assertRaises(JevError):
            _parse_response({"answers": {"a": {"type": "boolean", "probability": 0.7}}},
                            self.questions, 1.0)

    def test_a_response_without_answers_is_an_error(self):
        with self.assertRaises(JevError):
            _parse_response({"ok": True}, self.questions, 1.0)


class GatewayJevTest(unittest.TestCase):
    def test_a_missing_key_fails_before_any_request(self):
        with self.assertRaises(JevError):
            GatewayJev(api_key="")

    def test_empty_question_set_is_refused(self):
        with self.assertRaises(JevError):
            GatewayJev(api_key="k").evaluate(state={}, questions={})


class MockJevTest(unittest.TestCase):
    @staticmethod
    def _handler(question: str, state: Mapping[str, Any], options: Sequence[str]) -> Decision:
        return Decision(options[0], {o: 1.0 if i == 0 else 0.0 for i, o in enumerate(options)})

    def test_convenience_wrappers_route_through_evaluate(self):
        jev = MockJev(self._handler)
        self.assertEqual(jev.choose(question="Which?", state={}, options=["a", "b"]).choice, "a")
        self.assertEqual(jev.yes_no(question="Is it?", state={}).choice, "yes")
        self.assertEqual(jev.batches, [["q"], ["q"]])

    def test_duplicate_options_are_refused(self):
        with self.assertRaises(JevError):
            MockJev(self._handler).choose(question="Which?", state={}, options=["a", "a"])

    def test_an_answer_outside_the_option_set_is_refused(self):
        jev = MockJev(lambda q, s, o: Decision("made-up", {"made-up": 1.0}))
        with self.assertRaises(JevError):
            jev.choose(question="Which?", state={}, options=["a", "b"])


class Sam31GrounderTest(unittest.TestCase):
    def test_a_missing_key_fails_before_any_request(self):
        with self.assertRaises(GroundingError):
            Sam31Grounder(api_key="")

    def test_segmentation_still_requires_a_frame(self):
        self.assertTrue(Sam31Grounder(api_key="k").needs_frame)
        with self.assertRaises(GroundingError):
            Sam31Grounder(api_key="k").ground("the red button", None)

    def test_weak_detections_are_dropped_and_the_rest_ranked(self):
        grounder = Sam31Grounder(api_key="k", min_score=0.5, max_candidates=2)
        ranked = grounder._rank(
            [
                Candidate("a", "a", score=0.4),
                Candidate("b", "b", score=0.9),
                Candidate("c", "c", score=0.6),
                Candidate("d", "d", score=0.7),
            ]
        )
        self.assertEqual([c.ref for c in ranked], ["b", "d"])

    def test_boxes_survive_either_field_name(self):
        parsed = _parse_candidates(
            {"objects": [{"id": "1", "bbox": [0, 0, 1, 1], "score": 0.9}, {"box": [0, 0, 2, 2]}]},
            "the button",
        )
        self.assertEqual(parsed[0].box, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(parsed[1].label, "the button")
        self.assertIsNone(parsed[1].score)

    def test_a_response_that_is_not_an_object_is_an_error(self):
        with self.assertRaises(GroundingError):
            _parse_candidates(["nope"], "the button")


class FrameTest(unittest.TestCase):
    def test_frames_default_to_png(self):
        self.assertEqual(Frame(image=b"x").media_type, "image/png")


if __name__ == "__main__":
    unittest.main()
