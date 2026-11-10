import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from coach.conversation import SEP, parse_reply  # noqa: E402


class ParseTest(unittest.TestCase):
    def test_split(self):
        r, f = parse_reply(f"Nice! What do you do?\n{SEP}\n- 良いですね")
        self.assertEqual(r, "Nice! What do you do?")
        self.assertEqual(f, "- 良いですね")

    def test_no_separator(self):
        self.assertEqual(parse_reply("Hello"), ("Hello", ""))


if __name__ == "__main__":
    unittest.main()
