import os
import shutil
import time
from datetime import datetime
import unittest
import sys
from _setup_local import config
sys.path.append("../")
from dorc.device import all_devices, useable_devices
from dorc.trainer import Trainer


class SubTrainer(Trainer):
    def __init__(self, _cuda, *args, **kwargs):
        self._cuda = _cuda
        super().__init__(*args, **kwargs)

    @property
    def have_cuda(self):
        return self._cuda


class TrainerTestProps(unittest.TestCase):
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
        cls.trainer = SubTrainer(False, **cls.params)
        cls.trainer.reserved_gpus = []
        cls.trainer.reserve_gpus = lambda x: [True, None]
        cls.trainer._trainer_params["cuda"] = True

    def test_props_all_tests_present(self):
        props = self.trainer.props
        funcnames = self.__class__.__dict__.keys()
        for prop in props:
            with self.subTest(i=prop):
                self.assertIn("test_trainer_property_" + prop, funcnames)

    def test_trainer_property_version(self):
        pass

    def test_trainer_property_unique_id(self):
        pass

    def test_trainer_property_current_state(self):
        pass

    def test_trainer_property_loop_type(self):
        pass

    def test_trainer_property_logger(self):
        pass

    def test_trainer_property_logfile(self):
        pass

    def test_trainer_property_saves(self):
        pass

    def test_trainer_property_gpus(self):
        pass

    def test_trainer_property_system_info(self):
        pass

    def test_trainer_property_devices(self):
        pass

    def test_trainer_property_models(self):
        self.assertIsInstance(self.trainer.models, list)
        self.assertTrue(all([isinstance(x, list) for x in self.trainer.models]))

    def test_trainer_property_train_step_func(self):
        pass

    def test_trainer_property_val_step_func(self):
        pass

    def test_trainer_property_test_step_func(self):
        pass

    def test_trainer_property_active_model(self):
        pass

    def test_trainer_property_epoch(self):
        pass

    def test_trainer_property_max_epochs(self):
        pass

    def test_trainer_property_iterations(self):
        pass

    def test_trainer_property_max_iterations(self):
        pass

    def test_trainer_property_controls(self):
        pass

    def test_trainer_property_methods(self):
        pass

    def test_trainer_property_all_props(self):
        pass

    def test_trainer_property_extras(self):
        pass

    def test_trainer_property_running(self):
        pass

    def test_trainer_property_current_run(self):
        pass

    def test_trainer_property_paused(self):
        pass

    def test_trainer_property_adhoc_paused(self):
        pass

    def test_trainer_property_adhoc_aborted(self):
        pass

    def test_trainer_property_userfunc_paused(self):
        pass

    def test_trainer_property_userfunc_aborted(self):
        pass

    def test_trainer_property_best_save(self):
        pass

    def test_trainer_property_trainer_params(self):
        pass

    def test_trainer_property_model_params(self):
        pass

    def test_trainer_property_dataloader_params(self):
        pass

    def test_trainer_property_data(self):
        pass

    def test_trainer_property_updatable_params(self):
        pass

    def test_trainer_property_all_params(self):
        pass

    def test_trainer_property_train_losses(self):
        pass

    def test_trainer_property_progress(self):
        pass

    def test_trainer_property_user_funcs(self):
        pass

    def test_trainer_property_metrics(self):
        pass

    def test_trainer_property_val_samples(self):
        pass

    def test_trainer_property_all_post_epoch_hooks(self):
        pass

    def test_trainer_property_post_epoch_hooks_to_run(self):
        pass

    def test_trainer_property_items_to_log_dict(self):
        pass

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
