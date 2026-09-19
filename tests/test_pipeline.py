import unittest
from typing import Any, Mapping, Sequence

from jev_voice_cv.executor import Dedupe, DryRunExecutor
from jev_voice_cv.grounding import Frame
from jev_voice_cv.jev import MockJev
from jev_voice_cv.pipeline import NO_ACTION, VoicePipeline
from jev_voice_cv.schema import ActionSpec, Candidate, Decision, PlanKind

FRAME = Frame(image=b"", width=100, height=100)
SCROLL = ActionSpec("scroll", read_only=True, idempotent=True,
                    execute_threshold=0.70, speculate_threshold=0.90, needs_target=True)
CLICK = ActionSpec("click", execute_threshold=0.85, speculate_threshold=0.97, needs_target=True)
OPEN = ActionSpec("open_app", read_only=True, idempotent=True,
                  execute_threshold=0.75, speculate_threshold=0.90)


def flat(choice: str, confidence: float, options: Sequence[str]) -> Decision:
    rest = (1 - confidence) / max(len(options) - 1, 1)
    return Decision(choice, {o: confidence if o == choice else rest for o in options})


class FakeGrounder:
    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self.candidates = list(candidates)
        self.calls = 0

    def ground(self, prompt: str, frame: Frame) -> Sequence[Candidate]:
        self.calls += 1
        return self.candidates


def scripted(intent: str, intent_p: float, target_p: float = 0.9, addressed: float = 0.95):
    def handler(question: str, state: Mapping[str, Any], options: Sequence[str]) -> Decision:
        if list(options) == ["yes", "no"]:
            return Decision("yes", {"yes": addressed, "no": 1 - addressed})
        if "candidate" in question.lower():
            return flat(options[0], target_p, options)
        return flat(intent, intent_p, options)

    return handler


class SupersedeTest(unittest.TestCase):
    def test_result_for_an_older_transcript_is_discarded(self):
        pipeline = VoicePipeline(actions=[OPEN], jev=MockJev(scripted("open_app", 0.99)))
        stale = pipeline.submit("open the not", final=False)
        pipeline.submit("open the notes app", final=True)
        self.assertIs(pipeline.resolve(stale).kind, PlanKind.SUPERSEDED)

    def test_audio_arriving_mid_decision_cancels_that_decision(self):
        pipeline: VoicePipeline

        def handler(question, state, options):
            if list(options) == ["yes", "no"]:
                # New speech lands while Jev is still answering.
                pipeline.submit("open the notes app instead", final=False)
                return Decision("yes", {"yes": 0.99, "no": 0.01})
            raise AssertionError("must not keep scoring a superseded transcript")

        pipeline = VoicePipeline(actions=[OPEN], jev=MockJev(handler))
        ticket = pipeline.submit("open the notes app", final=True)
        self.assertIs(pipeline.resolve(ticket).kind, PlanKind.SUPERSEDED)

    def test_cancel_drops_everything_in_flight(self):
        pipeline = VoicePipeline(actions=[OPEN], jev=MockJev(scripted("open_app", 0.99)))
        ticket = pipeline.submit("open the notes app", final=True)
        pipeline.cancel()
        self.assertIs(pipeline.resolve(ticket).kind, PlanKind.SUPERSEDED)


class CommandFilterTest(unittest.TestCase):
    def test_speech_not_addressed_to_the_computer_is_rejected(self):
        pipeline = VoicePipeline(
            actions=[OPEN], jev=MockJev(scripted("open_app", 0.99, addressed=0.10))
        )
        plan = pipeline.resolve(pipeline.submit("I told him to open a ticket", final=True))
        self.assertIs(plan.kind, PlanKind.REJECT)
        self.assertEqual(plan.action, None)

    def test_filter_can_be_turned_off(self):
        pipeline = VoicePipeline(
            actions=[OPEN],
            jev=MockJev(scripted("open_app", 0.99, addressed=0.10)),
            command_filter=False,
        )
        self.assertIs(pipeline.resolve(pipeline.submit("open it", final=True)).kind, PlanKind.EXECUTE)


class GroundingTest(unittest.TestCase):
    def test_single_candidate_carries_the_detector_score(self):
        grounder = FakeGrounder([Candidate("btn-1", "red button", score=0.80)])
        pipeline = VoicePipeline(
            actions=[SCROLL], jev=MockJev(scripted("scroll", 0.95)), grounder=grounder
        )
        plan = pipeline.resolve(pipeline.submit("scroll to the red button", final=True, frame=FRAME))
        self.assertIs(plan.kind, PlanKind.EXECUTE)
        self.assertAlmostEqual(plan.confidence, 0.95 * 0.80)

    def test_ambiguous_candidates_go_to_jev_and_multiply_through(self):
        grounder = FakeGrounder(
            [Candidate("btn-1", "red submit", score=0.9), Candidate("btn-2", "red delete", score=0.9)]
        )
        jev = MockJev(scripted("click", 0.95, target_p=0.88))
        pipeline = VoicePipeline(actions=[CLICK], jev=jev, grounder=grounder)
        plan = pipeline.resolve(pipeline.submit("click the red one", final=True, frame=FRAME))
        self.assertEqual(plan.target.ref, "btn-1")
        self.assertAlmostEqual(plan.confidence, 0.95 * 0.88)
        # 0.836 sits under click's 0.85 bar, so it asks rather than clicking.
        self.assertIs(plan.kind, PlanKind.CONFIRM)

    def test_nothing_on_screen_matches(self):
        pipeline = VoicePipeline(
            actions=[SCROLL], jev=MockJev(scripted("scroll", 0.99)), grounder=FakeGrounder([])
        )
        final = pipeline.resolve(pipeline.submit("scroll to the bicycle", final=True, frame=FRAME))
        self.assertIs(final.kind, PlanKind.CONFIRM)
        self.assertEqual(final.reason, "no grounded target")
        partial = pipeline.resolve(pipeline.submit("scroll to the bi", final=False, frame=FRAME))
        self.assertIs(partial.kind, PlanKind.HOLD)

    def test_targeted_action_without_a_frame_does_not_reach_the_grounder(self):
        grounder = FakeGrounder([Candidate("btn-1", "red button")])
        pipeline = VoicePipeline(
            actions=[SCROLL], jev=MockJev(scripted("scroll", 0.99)), grounder=grounder
        )
        plan = pipeline.resolve(pipeline.submit("scroll to the red button", final=True))
        self.assertIs(plan.kind, PlanKind.CONFIRM)
        self.assertEqual(grounder.calls, 0)


class IntentTest(unittest.TestCase):
    def test_no_action_option_keeps_unmatched_speech_out_of_the_action_set(self):
        pipeline = VoicePipeline(actions=[OPEN], jev=MockJev(scripted(NO_ACTION, 0.9)))
        plan = pipeline.resolve(pipeline.submit("what time is it", final=True))
        self.assertIs(plan.kind, PlanKind.REJECT)

    def test_reserved_option_name_cannot_be_an_action(self):
        with self.assertRaises(ValueError):
            VoicePipeline(actions=[ActionSpec(NO_ACTION)], jev=MockJev(scripted("x", 1.0)))


class ExecutorTest(unittest.TestCase):
    def test_speculated_action_is_not_run_twice_when_the_final_confirms_it(self):
        clock = iter([0.0, 0.5, 10.0])
        executor = DryRunExecutor(dedupe=Dedupe(window_seconds=3.0, clock=lambda: next(clock)))
        pipeline = VoicePipeline(actions=[OPEN], jev=MockJev(scripted("open_app", 0.99)))
        speculated = pipeline.resolve(pipeline.submit("open the notes app", final=False))
        confirmed = pipeline.resolve(pipeline.submit("open the notes app", final=True))
        self.assertIs(speculated.kind, PlanKind.SPECULATE)
        executor.run(speculated)
        executor.run(confirmed)
        executor.run(confirmed)  # clock moved past the window
        self.assertEqual(
            executor.log,
            [
                "speculate: open_app (p=0.99)",
                "skip (already ran): open_app",
                "execute: open_app (p=0.99)",
            ],
        )

    def test_non_runnable_plan_is_refused(self):
        pipeline = VoicePipeline(actions=[OPEN], jev=MockJev(scripted(NO_ACTION, 0.9)))
        plan = pipeline.resolve(pipeline.submit("hello", final=True))
        with self.assertRaises(ValueError):
            DryRunExecutor().run(plan)
