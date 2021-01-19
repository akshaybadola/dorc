import os
import shutil
import pytest
import torch
from datetime import datetime
import unittest
import sys
from pydantic import ValidationError
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


@pytest.mark.quick
class TrainerTestDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup a simple trainer with MNIST dataset."""
        import importlib
        import _setup_local as setup
        importlib.reload(setup)
        config = setup.config
        cls.config = config
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")
        time_str = datetime.now().isoformat()
        os.mkdir(f".test_dir/test_session/{time_str}")
        cls.data_dir = f".test_dir/test_session/{time_str}"
        cls.params = {"data_dir": cls.data_dir, **cls.config}

    def setUp(self):
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer.trainer_params.cuda = True

    def test_check_trainer_bad_params_gpus(self):
        cases = [None, "bleh", -2, False, {"bleh"}, ["test"]]
        for i, case in enumerate(cases):
            with self.subTest(i=i):
                if case == "bleh" or case == {"bleh"} or case == ["test"]:
                    with pytest.raises(ValidationError):
                        self.trainer.trainer_params.gpus = case
                self.assertEqual(self.trainer.gpus, [])

    def test_check_trainer_good_params_gpus(self):
        cases = [0, 4, [0, 1], [-1]]
        for i, case in enumerate(cases):
            with self.subTest(i=i):
                self.trainer.trainer_params.gpus = case
                self.assertEqual(self.trainer.gpus,
                                 case if isinstance(case, list) else [case])

    def test_allocate_devices(self):
        # 1. explicitly mentioned gpus are given preference
        # 2. after that auto
        # 3. after that parallel
        # 4. What about distributed?
        pass

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_check_trainer_set_gpus(self):
        have_gpus = all_devices()
        useable_gpus = useable_devices()
        if have_gpus != useable_gpus:
            self.trainer.gpus = have_gpus
            with self.subTest(i="some_gpus_not_supported"):
                self.trainer._maybe_init_gpus()
                self.assertEqual(self.trainer.gpus, useable_gpus)
        with self.subTest(i="more_gpus_than_available_given"):
            self.trainer.reserved_gpus = []
            self.trainer.gpus = have_gpus + [*range(max(have_gpus) + 1, max(have_gpus) + 3)]
            self.trainer._maybe_init_gpus()
            self.assertEqual(self.trainer.gpus, useable_gpus)

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_check_trainer_set_device_one_gpu_no_cuda_given_AND_gpus_given(self):
        self.trainer.trainer_params.cuda = False
        gpus = {"one_gpu": [0], "two_gpus": [0, 1]}
        for i in gpus:
            if i == "two_gpus" and len(all_devices()) < 2:
                continue
            with self.subTest(i=i):
                self.trainer.trainer_params.gpus = gpus[i]
                self.trainer._maybe_init_gpus()
                self.assertEqual(self.trainer.gpus, gpus[i])
                self.trainer._set_device()
                self.assertEqual(self.trainer.gpus, [-1])
                self.trainer._init_models()
                self.assertEqual(self.trainer._models["net"]._device, torch.device("cpu"))

    def test_check_trainer_set_device_cuda_given_AND_no_gpus_given(self):
        self.trainer.trainer_params.gpus = []
        self.trainer._maybe_init_gpus()
        self.trainer._set_device()
        self.assertEqual(self.trainer.gpus, [-1])
        self.trainer._init_models()
        self.assertEqual(self.trainer._models["net"]._device, torch.device("cpu"))

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_check_trainer_set_device_one_gpu_one_model(self):
        # if one gpu and one model, set model._device even when not specified in
        # params? when have_cuda and cuda_given
        self.trainer.trainer_params.gpus = [0]
        self.trainer._maybe_init_gpus()
        self.trainer._set_device()
        self.assertEqual(self.trainer.gpus, [0])
        self.trainer._init_models()
        self.assertEqual(self.trainer._models["net"]._device, torch.device(0))

    @unittest.skipIf(len(all_devices()) < 2, f"Cannot run without at least 2 GPUs.")
    def test_check_trainer_set_device_many_gpus_one_model(self):
        self.trainer.trainer_params.gpus = [0, 1]
        self.trainer._maybe_init_gpus()
        self.trainer._set_device()
        self.assertEqual(self.trainer.gpus, [0, 1])
        self.trainer._init_models()
        self.assertEqual(self.trainer._models["net"]._device, "dataparallel")
        with self.subTest(i="tensor_is_placed_correctly_on_model_device"):
            import torch
            bleh = torch.rand(100, 100)
            self.assertEqual(self.trainer._models["net"].to(bleh).device,
                             torch.device(self.trainer._models["net"].gpus[0]))

    def test_check_trainer_set_device_one_gpu_many_models(self):
        pass

    def test_check_trainer_set_device_many_gpus_many_models(self):
        pass

    # NOTE: This should test correct allocation of devices across many gpus and
    #       perhaps DDP
    def test_check_trainer_set_device_load_unload_auto_adjust(self):
        pass

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
