import unittest

from jev_voice_cv.policy import combine, gate
from jev_voice_cv.schema import ActionSpec, Candidate, PlanKind

READ = ActionSpec("scroll", read_only=True, idempotent=True,
                  execute_threshold=0.70, speculate_threshold=0.90)
WRITE = ActionSpec("type_text", execute_threshold=0.85, speculate_threshold=0.97)
DESTRUCTIVE = ActionSpec("delete", always_confirm=True, execute_threshold=0.95,
                         speculate_threshold=0.99, needs_target=True)
TARGET = Candidate("btn-1", "red button")


class GateTest(unittest.TestCase):
    def test_partial_read_only_speculates_above_its_higher_bar(self):
        self.assertIs(gate(READ, 0.95, final=False).kind, PlanKind.SPECULATE)

    def test_partial_read_only_holds_between_the_two_bars(self):
        # Above the end-of-turn bar, still below the speculation bar.
        plan = gate(READ, 0.80, final=False)
        self.assertIs(plan.kind, PlanKind.HOLD)
        self.assertIs(gate(READ, 0.80, final=True).kind, PlanKind.EXECUTE)

    def test_partial_never_speculates_a_write_however_confident(self):
        self.assertIs(gate(WRITE, 0.999, final=False).kind, PlanKind.HOLD)

    def test_final_below_execute_threshold_asks(self):
        self.assertIs(gate(WRITE, 0.84, final=True).kind, PlanKind.CONFIRM)
        self.assertIs(gate(WRITE, 0.86, final=True).kind, PlanKind.EXECUTE)

    def test_destructive_action_asks_at_any_confidence(self):
        plan = gate(DESTRUCTIVE, 1.0, final=True, target=TARGET)
        self.assertIs(plan.kind, PlanKind.CONFIRM)

    def test_below_reject_floor_is_rejected_on_final_and_held_on_partial(self):
        self.assertIs(gate(WRITE, 0.2, final=True).kind, PlanKind.REJECT)
        self.assertIs(gate(WRITE, 0.2, final=False).kind, PlanKind.HOLD)

    def test_missing_target_never_falls_through_to_a_guess(self):
        self.assertIs(gate(DESTRUCTIVE, 0.99, final=True, target=None).kind, PlanKind.CONFIRM)
        self.assertIs(gate(DESTRUCTIVE, 0.99, final=False, target=None).kind, PlanKind.HOLD)

    def test_text_argument_is_flagged_as_not_coming_from_jev(self):
        self.assertEqual(
            gate(ActionSpec("note", needs_text_arg=True), 0.9, final=True).text_arg_from, "llm"
        )


class CombineTest(unittest.TestCase):
    def test_joint_confidence_is_the_product(self):
        self.assertAlmostEqual(combine(0.9, 0.8), 0.72)

    def test_targetless_action_keeps_intent_confidence(self):
        self.assertAlmostEqual(combine(0.9, None), 0.9)

    def test_weak_target_drags_a_confident_intent_below_the_bar(self):
        self.assertIs(gate(WRITE, combine(0.95, 0.60), final=True).kind, PlanKind.CONFIRM)


class ActionSpecTest(unittest.TestCase):
    def test_speculation_bar_must_not_sit_below_the_execution_bar(self):
        with self.assertRaises(ValueError):
            ActionSpec("bad", execute_threshold=0.9, speculate_threshold=0.5)

    def test_read_only_action_needs_a_strictly_higher_speculation_bar(self):
        with self.assertRaises(ValueError):
            ActionSpec("bad", read_only=True, idempotent=True,
                       execute_threshold=0.9, speculate_threshold=0.9)
