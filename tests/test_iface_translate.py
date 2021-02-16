import pytest
import sys
sys.path.append("..")
from dorc.mods import load_module_exports, eval_python_exprs
from dorc.interfaces.translation import TranslationLayer
from dorc.trainer import config as trainer_config


@pytest.mark.quick
def test_translate_config_from_json(json_config):
    config = json_config
    tlayer = TranslationLayer(config, "")
    test_config = tlayer.from_json()
    trainer_config.Config(**test_config)


@pytest.mark.quick
def test_translate_to_json(json_config):
    config = json_config
    tlayer = TranslationLayer(config, "")
    test_config = tlayer.from_json()
    config = trainer_config.Config(**test_config)
    config.schema()
    adam = config.optimizers["Adam"]
    tlayer.to_json(adam)
    tlayer.to_json(config)


@pytest.mark.todo
@pytest.mark.quick
def test_translate_to_json_function(json_config):
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_translate_to_json_module(json_config):
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_translate_to_json_class(json_config):
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_translate_to_json_basemodel(json_config):
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_translate_to_json_numpy_tensor(json_config):
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_translate_to_json_class_callable(json_config):
    pass
