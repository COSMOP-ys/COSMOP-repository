import unittest

from jev_voice_cv.grounding import Frame, Sam31Grounder, StaticGrounder, _parse_candidates
from jev_voice_cv.jev import JevError, MockJev, OpenRouterJev, _parse_probabilities
from jev_voice_cv.schema import Candidate, Decision

FRAME = Frame(image=b"")


class ParseProbabilitiesTest(unittest.TestCase):
    def test_probability_map(self):
        self.assertEqual(
            _parse_probabilities({"decision": {"probabilities": {"a": 0.7, "b": 0.3}}}),
            {"a": 0.7, "b": 0.3},
        )

    def test_top_level_distribution(self):
        self.assertEqual(_parse_probabilities({"distribution": {"a": 1.0}}), {"a": 1.0})

    def test_choice_plus_confidence(self):
        self.assertEqual(
            _parse_probabilities({"decision": {"choice": "b", "confidence": 0.82}}), {"b": 0.82}
        )

    def test_unrecognised_shape_reports_the_payload(self):
        with self.assertRaises(JevError) as caught:
            _parse_probabilities({"decision": {"text": "maybe"}})
        self.assertEqual(caught.exception.payload, {"decision": {"text": "maybe"}})


class JevClientTest(unittest.TestCase):
    def test_missing_key_fails_before_any_request(self):
        with self.assertRaises(JevError):
            OpenRouterJev(api_key="")

    def test_option_set_is_validated_locally(self):
        client = OpenRouterJev(api_key="test")
        with self.assertRaises(JevError):
            client.choose(question="q", state={}, options=[])
        with self.assertRaises(JevError):
            client.choose(question="q", state={}, options=[f"o{i}" for i in range(256)])
        with self.assertRaises(JevError):
            client.choose(question="q", state={}, options=["a", "a"])

    def test_mock_cannot_answer_outside_the_schema(self):
        mock = MockJev(lambda q, s, o: Decision("elsewhere", {"elsewhere": 1.0}))
        with self.assertRaises(JevError):
            mock.choose(question="q", state={}, options=["a", "b"])


class ParseCandidatesTest(unittest.TestCase):
    def test_boxes_scores_and_labels(self):
        payload = {
            "objects": [
                {"id": "o1", "label": "red bicycle", "box": [0.1, 0.2, 0.3, 0.4], "score": 0.9},
                {"bbox": [0, 0, 1, 1]},
            ]
        }
        first, second = _parse_candidates(payload, "red bicycle")
        self.assertEqual((first.ref, first.label, first.score), ("o1", "red bicycle", 0.9))
        self.assertEqual(first.box, (0.1, 0.2, 0.3, 0.4))
        # Missing id/label fall back to the index and the prompt.
        self.assertEqual((second.ref, second.label, second.score), ("obj-1", "red bicycle", None))

    def test_empty_response_grounds_nothing(self):
        self.assertEqual(_parse_candidates({}, "anything"), [])


class GrounderTest(unittest.TestCase):
    def test_low_scoring_detections_are_dropped_and_the_rest_ranked(self):
        grounder = Sam31Grounder(api_key="test", min_score=0.5, max_candidates=2)
        payload = {
            "objects": [
                {"id": "weak", "score": 0.2},
                {"id": "mid", "score": 0.6},
                {"id": "strong", "score": 0.95},
                {"id": "also", "score": 0.7},
            ]
        }
        kept = [c.ref for c in grounder._rank(_parse_candidates(payload, "x"))]
        self.assertEqual(kept, ["strong", "also"])

    def test_static_grounder_matches_on_label_words(self):
        grounder = StaticGrounder([Candidate("a", "red submit button"), Candidate("b", "sidebar")])
        self.assertEqual([c.ref for c in grounder.ground("click the submit button", FRAME)], ["a"])
        self.assertEqual(grounder.ground("click the bicycle", FRAME), [])
