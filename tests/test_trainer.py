import os
import shutil
from datetime import datetime
import unittest
import sys
from _setup import config
sys.path.append("../")
from trainer.trainer import Trainer


class TrainerTest(unittest.TestCase):
    def setUp(self):
        """Setup a simple trainer with MNIST dataset."""
        self.config = config
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")
        time_str = datetime.now().isoformat()
        os.mkdir(f".test_dir/test_session/{time_str}")
        self.data_dir = f".test_dir/test_session/{time_str}"
        self.trainer = Trainer(**{"data_dir": self.data_dir, **self.config})

    # TODO: Tweak config's various parameters and check for errors
    #       Would have to be subtests.
    #
    # 1. {val,test}_loaders are None
    # 2. train_loader None should fail init
    # 3. dataloader as (function, params) pair should pass but sub cases should be handled
    # 4. uid and session creation
    # 5. criteria should be callable and non callable should fail
    #     None params shouldn't be allowed
    # 6. Can extra_metrics be None? Or should it be empty dict?
    #     Should we replace all None's with empty {}?
    # 7. various thingies in trainer_params
    # 8. Different training_steps than "train", "val", "test"?
    # 9. data = None and such combinations
    # 10. model_params cannot be None?
    # 11. model has to be in model_defs
    # 12. update_function checks
    # 13. Test all extras, helpers, return values also
    # 14. Test controls, return values also
    # 15. All stateless funcs
    # 16. Fixes and tests for broken funcs
    def test_trainer_init(self):
        self.trainer._init_all()
        self.assertFalse(self.trainer._have_resumed)
        self.assertTrue(self.trainer.paused)

    def test_trainer_resume_force(self):
        pass

    # NOTE: All of these tests should be run with various params
    # def test_trainer_save(self):
    #     pass

    # with single gpu do a bunch of things
    # def test_single_gpu(self):
    #     pass

    # def test_dataparallel(self):
    #     pass

    # def test_iterations_only(self):
    #     pass

    # Check with various data params and configs
    # def test_post_epoch_hooks(self):
    #     pass

    # What if they don't have certain keys?
    # def test_update_funcs(self):
    #     pass

    # update module also
    # def test_add_module(self):
    #     pass

    # def test_device_logging(self):
    #     pass

    def test_load_saves(self):
        data = {}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] Missing params \"weights\""))
        data = {"weights": "meh"}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] Invalid or no such method"))
        data = {"weights": "meh", "method": "load"}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] No such file"))

    # Need to test
    # trainer setup has to be tested separately
    # multi_models, multi_criteria like setup
    # sampling
    # recurrent and other such models
    # rest of the functions
    # user func transitions
    def tearDown(self):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
