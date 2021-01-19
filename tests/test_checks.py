import pytest
import unittest
import sys
sys.path.append("../")
from trainer.checks import Checks


@pytest.mark.quick
class ChecksTest(unittest.TestCase):
    def setUp(self):
        self.checks = Checks(print, print)
        self.checks.add(4 > 5, "4 > 5, Should be false")
        self.checks.add(6 > 5, "6 > 5, Should be True")
        self.checks.check_all_true()
        with self.checks.catch_and_log("yay", "no") as check_true:
            if check_true:
                print("test")

    def test_first_predicate_false(self):
        self.assertFalse(self.checks.status)
        self.assertEqual(self.checks.message, "4 > 5, Should be false")

    def test_second_predicate_false(self):
        self.checks.clear()
        self.checks.add(6 > 5, "6 > 5, Should be True")
        self.checks.add(4 > 5, "4 > 5, Should be false")
        self.checks.check_all_true()
        with self.checks.catch_and_log("yay", "no") as check_true:
            if check_true:
                print("test")
        self.assertFalse(self.checks.status)
        self.assertEqual(self.checks.message, "4 > 5, Should be false")

    def test_all_predicates_true(self):
        self.checks.clear()
        self.checks.add(4 < 5, "4 < 5, Should be True")
        self.checks.add(6 > 5, "6 > 5, Should be True")
        self.checks.check_all_true()
        self.assertTrue(self.checks.status)
        self.assertEqual(self.checks.message, "All checks passed")
        with self.checks.catch_and_log("yay", "no") as check_true:
            if check_true:
                raise Exception("meh")
        self.assertFalse(self.checks.status)
        self.assertEqual(self.checks.message, "no Error occured. Exception: meh")

    def test_clear(self):
        self.checks.clear()
        self.assertIsNone(self.checks.status)
        self.assertIsNone(self.checks.message)
        self.assertEqual(self.checks._list, [])

    def test_one_predicate_true(self):
        self.checks.clear()
        self.checks.add(4 < 5, "4 < 5, Should be Tue")
        self.checks.add(6 > 5, "6 > 5, Should be True")
        self.checks.check_all_true()


if __name__ == '__main__':
    unittest.main()
