import pytest
from dorc.trainer import funcs


@pytest.mark.quick
def test_trainer_hooks_load_funcs(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()         # should init hooks
    all_funcs = {*trainer.funcs, *trainer.funcs_with_args}
    assert all([x in all_funcs for x, y in funcs.__dict__.items()
                if not x.startswith("_") and callable(y) and "hook" in x])


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_hooks_run_post_epoch_permanent_items(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()
    trainer._run_post_epoch_hook()


@pytest.mark.quick
def test_trainer_hooks_add_to_post_epoch_hook(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()
    # pytest.set_trace()
    # add hook by user, from source or available module
    # ensure it's run in next loop if added


@pytest.mark.quick
@pytest.mark.todo
def test_trainer_hooks_delete_from_post_epoch_hook(params_and_trainer):
    params, trainer = params_and_trainer
    # delete hook and ensure it's deleted from hooks_to_run
    # and doesn't run on next loop


@pytest.mark.quick
@pytest.mark.todo
def test_trainer_hooks_modify_post_epoch_hook(params_and_trainer):
    params, trainer = params_and_trainer
    # adding or deleting hook ensures it runs or doesn't run in next loop


@pytest.mark.quick
@pytest.mark.todo
def test_trainer_hooks_dump_and_restore(params_and_trainer):
    params, trainer = params_and_trainer
    # as they're stateless functions they should be dumpable
    # adding or deleting hook ensures it runs or doesn't run in next loop
