import pytest
import pathlib
from fixtures import *


# def pytest_collection_modifyitems(config, items):
#     rootdir = pathlib.Path(config.rootdir)
#     for item in items:
#         rel_path = pathlib.Path(item.fspath).relative_to(rootdir)
#         mark_name = next((part for part in rel_path.parts if part.endswith('_tests')), '').rstrip('_tests')
#         if item.own_markers:
#             pytest.set_trace()
#         if mark_name:
#             mark = getattr(pytest.mark, mark_name)
#             item.add_marker(mark)
