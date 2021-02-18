import pytest
import sys
from dorc.__main__ import load_or_create_config


@pytest.mark.todo
def test_read_config():
    opts = load_or_create_config([])
    assert hasattr(opts, "config_file")
    assert opts.root_dir == "/home/joe/.dorc/"
