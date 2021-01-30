import pytest
import sys
sys.path.append("..")
from trainer.mods import load_module_exports, eval_python_exprs
from trainer.interfaces.translation import TranslationLayer
from trainer.trainer import config as trainer_config


@pytest.mark.quick
def test_translate_config_from_json(json_config):
    config = json_config
    tlayer = TranslationLayer(config)
    test_config = tlayer.from_json()
    trainer_config.Config(**test_config)


@pytest.mark.quick
def test_translate_to_json(json_config):
    config = json_config
    tlayer = TranslationLayer(config)
    test_config = tlayer.from_json()
    config = trainer_config.Config(**test_config)
    config.schema()
    adam = config.optimizers["Adam"]
    tlayer.to_json(adam)
