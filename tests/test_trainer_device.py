import os
import shutil
import torch
from datetime import datetime
import unittest
import sys
from _setup_local import config
sys.path.append("../")
from trainer.device import all_devices, useable_devices
from trainer.trainer import Trainer


class SubTrainer(Trainer):
    def __init__(self, _cuda, *args, **kwargs):
        self._cuda = _cuda
        super().__init__(*args, **kwargs)

    @property
    def have_cuda(self):
        return self._cuda


class TrainerTestDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup a simple trainer with MNIST dataset."""
        cls.config = config
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")
        time_str = datetime.now().isoformat()
        os.mkdir(f".test_dir/test_session/{time_str}")
        cls.data_dir = f".test_dir/test_session/{time_str}"
        cls.params = {"data_dir": cls.data_dir, **cls.config}

    def test_check_trainer_bad_params_gpus(self):
        self.trainer = SubTrainer(False, **self.params)
        cases = [None, "bleh", -2, False, {"bleh"}, ["test"]]
        for i, case in enumerate(cases):
            with self.subTest(i=i):
                self.trainer._trainer_params["gpus"] = case
                self.trainer._check_gpus_param()
                self.assertEqual(self.trainer._gpus, [-1])

    def test_check_trainer_good_params_gpus(self):
        self.trainer = SubTrainer(False, **self.params)
        cases = [0, 4, [0, 1], [-1]]
        for i, case in enumerate(cases):
            with self.subTest(i=i):
                self.trainer._trainer_params["gpus"] = case
                self.trainer._check_gpus_param()
                self.assertEqual(self.trainer._gpus,
                                 case if isinstance(case, list) else [case])

    def test_check_trainer_set_gpus(self):
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["cuda"] = True
        have_gpus = all_devices()
        useable_gpus = useable_devices()
        if have_gpus != useable_gpus:
            self.trainer._gpus = have_gpus
            with self.subTest(i="some_gpus_not_supported"):
                self.trainer._maybe_init_gpus()
                self.assertEqual(self.trainer._gpus, useable_gpus)
        with self.subTest(i="more_gpus_than_available_given"):
            self.trainer.reserved_gpus = []
            self.trainer._gpus = have_gpus + [*range(max(have_gpus) + 1, max(have_gpus) + 3)]
            self.trainer._maybe_init_gpus()
            self.assertEqual(self.trainer._gpus, useable_gpus)

    def test_check_trainer_set_device_one_gpu_no_cuda_given_AND_gpus_given(self):
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["cuda"] = False
        self.trainer._trainer_params["gpus"] = [0, 1]
        self.trainer._check_gpus_param()
        self.trainer._maybe_init_gpus()
        self.assertEqual(self.trainer._gpus, [0, 1])
        self.trainer._set_device()
        self.assertEqual(self.trainer._gpus, [-1])

    def test_check_trainer_set_device_cuda_given_AND_no_gpus_given(self):
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["cuda"] = True
        self.trainer._trainer_params["gpus"] = []
        self.trainer._check_gpus_param()
        self.trainer._maybe_init_gpus()
        self.trainer._set_device()
        self.assertEqual(self.trainer._gpus, [-1])

    def test_check_trainer_set_device_one_model(self):
        # if one gpu and one model, set model._device even when not specified in
        # params? when have_cuda and cuda_given
        self.trainer = SubTrainer(False, **self.params)
        with self.subTest(i="one_gpu"):
            self.trainer.reserved_gpus = []
            self.trainer.reserve_gpus = lambda x: [True, None]
            self.trainer._trainer_params["cuda"] = True
            self.trainer._trainer_params["gpus"] = [0]
            self.trainer._check_gpus_param()
            self.trainer._maybe_init_gpus()
            self.trainer._set_device()
            self.assertEqual(self.trainer._gpus, [0])
            self.trainer._init_models()
            self.assertEqual(self.trainer._models["net"]._device, torch.device(0))

    def test_check_trainer_set_device_one_gpu_many_models(self):
        pass

    def test_check_trainer_set_device_many_gpus_many_models(self):
        pass


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



    # def test_trainer_init(self):
    #     self.assertFalse(self.trainer._have_resumed)
    #     self.assertTrue(self.trainer.paused)

    # def test_trainer_log(self):
    #     self.trainer._init_all()
    #     self.trainer.start()
    #     time.sleep(5)
    #     import ipdb; ipdb.set_trace()

    # def test_trainer_resume_force(self):
    #     pass

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

    # def test_load_saves(self):
    #     data = {}
    #     self.assertEqual(self.trainer.load_saves(data),
    #                      (False, "[load_saves()] Missing params \"weights\""))
    #     data = {"weights": "meh"}
    #     self.assertEqual(self.trainer.load_saves(data),
    #                      (False, "[load_saves()] Invalid or no such method"))
    #     data = {"weights": "meh", "method": "load"}
    #     self.assertEqual(self.trainer.load_saves(data),
    #                      (False, "[load_saves()] No such file"))

    # Need to test
    # trainer setup has to be tested separately
    # multi_models, multi_criteria like setup
    # sampling
    # recurrent and other such models
    # rest of the functions
    # user func transitions
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
