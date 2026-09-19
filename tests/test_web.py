import unittest

from jev_voice_cv.schema import Candidate, Plan, PlanKind
from jev_voice_cv.web.server import OfflineChooser, Session, plan_to_json

NODES = [
    {"ref": "n0", "role": "button", "name": "Delete account", "attrs": {"colour": "red"}},
    {"ref": "n1", "role": "button", "name": "Save changes", "attrs": {"colour": "green"}},
    {"ref": "n2", "role": "textbox", "name": "Email", "attrs": {"placeholder": "you@example.com"}},
]


def resolve(session: Session, text: str, final: bool) -> dict:
    return session.resolve({"text": text, "final": final, "nodes": NODES})


class OfflineSessionTest(unittest.TestCase):
    """The stub is fiction, but the gate it feeds is the real one."""

    def setUp(self):
        self.session = Session(offline=True)

    def test_a_read_only_action_may_run_before_the_sentence_ends(self):
        plan = resolve(self.session, "scroll to save changes", False)
        self.assertEqual(plan["kind"], "speculate")
        self.assertEqual(plan["target"]["ref"], "n1")

    def test_a_write_action_waits_for_the_end_of_the_sentence(self):
        self.assertEqual(resolve(self.session, "click save changes", False)["kind"], "hold")
        self.assertEqual(resolve(self.session, "click save changes", True)["kind"], "execute")

    def test_a_destructive_action_asks_even_when_confident(self):
        plan = resolve(self.session, "delete the account", True)
        self.assertEqual(plan["kind"], "confirm")
        self.assertGreater(plan["confidence"], 0.9)
        self.assertFalse(plan["runnable"])

    def test_talk_about_the_computer_is_not_talk_to_it(self):
        plan = resolve(self.session, "anyway I told him to click save", True)
        self.assertEqual(plan["kind"], "reject")
        self.assertIsNone(plan["action"])

    def test_an_unmatched_description_never_produces_a_target(self):
        plan = resolve(self.session, "click the bicycle", True)
        self.assertIsNone(plan["target"])
        self.assertEqual(plan["reason"], "no grounded target")

    def test_the_snapshot_is_whatever_the_page_last_sent(self):
        self.session.resolve({"text": "click save changes", "final": True, "nodes": []})
        plan = self.session.resolve({"text": "click save changes", "final": True, "nodes": []})
        self.assertIsNone(plan["target"])

    def test_cancelling_strands_the_request_in_flight(self):
        ticket = self.session.pipeline.submit("click save changes", final=True)
        self.session.cancel()
        self.assertIs(self.session.pipeline.resolve(ticket).kind, PlanKind.SUPERSEDED)


class OfflineChooserTest(unittest.TestCase):
    def test_probabilities_form_a_distribution(self):
        from jev_voice_cv.jev import choice

        answers = OfflineChooser().evaluate(
            state={"transcript": "click the thing"},
            questions={"intent": choice("Which?", ["click_element", "open_app", "no_action"])},
        )
        total = sum(answers["intent"].probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertEqual(answers["intent"].choice, "click_element")

    def test_no_cue_falls_through_to_no_action(self):
        from jev_voice_cv.jev import choice

        answers = OfflineChooser().evaluate(
            state={"transcript": "what time is it"},
            questions={"intent": choice("Which?", ["click_element", "no_action"])},
        )
        self.assertEqual(answers["intent"].choice, "no_action")


class SerialisationTest(unittest.TestCase):
    def test_a_plan_survives_the_trip_to_the_page(self):
        plan = Plan(
            PlanKind.EXECUTE,
            action="click_element",
            target=Candidate("n1", "button \"Save\"", score=0.9),
            confidence=0.912345,
            reason="final transcript, above execute threshold",
            trace=("addressed=0.93", "intent=click_element@0.96"),
        )
        payload = plan_to_json(plan)
        self.assertEqual(payload["confidence"], 0.9123)
        self.assertTrue(payload["runnable"])
        self.assertEqual(payload["target"]["ref"], "n1")
        self.assertEqual(len(payload["trace"]), 2)

    def test_a_refusal_carries_no_target_and_is_not_runnable(self):
        payload = plan_to_json(Plan(PlanKind.REJECT, reason="not addressed to the computer"))
        self.assertFalse(payload["runnable"])
        self.assertIsNone(payload["target"])


if __name__ == "__main__":
    unittest.main()
