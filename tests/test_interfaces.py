import unittest
import sys
sys.path.append("../")
from trainer.interfaces import FlaskInterface
from trainer.trainer import Trainer


class InterfaceTest(unittest.TestCase):
    def setUp(self):
        # Don't need daemon for trainer interface actually
        pass

    # def test_trainer_interface(self):
    #     # what to test?
    #     pass

    # def test_compare_sessions(self):
    #     "compare metrics and other attributes between two sessions"
    #     pass


if __name__ == '__main__':
    unittest.main()
