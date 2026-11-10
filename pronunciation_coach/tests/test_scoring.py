import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from coach.scoring import align, mistakes, score  # noqa: E402


class ScoringTest(unittest.TestCase):
    def test_perfect(self):
        r = align(list("rait"), list("rait"))
        self.assertEqual(score(r), 100.0)
        self.assertEqual(mistakes(r), [])

    def test_substitution_l_r(self):
        # right -> light(r が l に)
        r = align(["r", "aɪ", "t"], ["l", "aɪ", "t"])
        self.assertEqual([x.status for x in r], ["substituted", "ok", "ok"])
        self.assertAlmostEqual(score(r), 66.7, places=1)
        self.assertEqual(mistakes(r), ["/r/ が /l/ に聞こえた"])

    def test_missing_and_extra(self):
        r = align(["s", "t", "ɑː", "p"], ["s", "ɑ", "p", "u"])  # 長音記号は無視される
        statuses = [x.status for x in r]
        self.assertIn("missing", statuses)
        self.assertIn("extra", statuses)

    def test_stress_ignored(self):
        r = align(["ˈhɛ", "loʊ"], ["hɛ", "loʊ"])
        self.assertEqual(score(r), 100.0)


if __name__ == "__main__":
    unittest.main()
