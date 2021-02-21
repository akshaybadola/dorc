import pytest
from dorc.helpers import Hook
from dorc.trainer import funcs


def dummy_func():
    return None


@pytest.mark.quick
def test_hook_should_not_modify_permanent_items():
    hook = Hook({"validate": funcs.validate_func, "test": funcs.test_func})
    with pytest.raises(ValueError):
        hook.remove("validate")
    with pytest.raises(ValueError):
        hook.append("validate", dummy_func)


@pytest.mark.quick
def test_hook_insert_delete():
    hook = Hook({"validate": funcs.validate_func, "test": funcs.test_func})
    hook.insert(0, "dummy", dummy_func)
    assert hook.index("dummy") == len(hook._permanent_items)
    hook.append("monkey", dummy_func)
    assert hook.index("monkey") == (len(hook) - 1)
    hook.append("dummy", dummy_func)
    assert hook.index("dummy") == (len(hook) - 1)
    hook.insert(-2, "last_but_one", dummy_func)
    assert hook.index("last_but_one") == 2
    hook.insert(-3, "last_but_one", dummy_func)
    assert hook.index("last_but_one") == 2
    hook.push("first", dummy_func)
    assert hook.index("first") == len(hook._permanent_items)


@pytest.mark.quick
def test_hook_iter_getitem():
    hook = Hook({"validate": funcs.validate_func, "test": funcs.test_func, "log": funcs.log_func})
    hook.insert(0, "dummy", dummy_func)
    assert all(isinstance(h, str) for h in hook)
    assert all(isinstance(h, str) for h in hook.keys())
    assert all(callable(h) for h in hook.values())
