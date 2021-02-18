import pytest


@pytest.mark.todo
def test_trainer_consistency(params_and_trainer):
    pass


# TODO
# def test_trainer_log():
#     trainer._init_all()
#     trainer.start()
#     time.sleep(5)
#     import ipdb; ipdb.set_trace()

# TODO
# def test_trainer_resume_force():
#     pass

# NOTE: All of these tests should be run with various params
# def test_trainer_save():
#     pass

# TODO
# def test_dataparallel():
#     pass

# TODO
# def test_iterations_only():
#     pass

# TODO
# Check with various data params and configs
# def test_post_epoch_hooks():
#     pass

# TODO
# What if they don't have certain keys?
# def test_update_funcs():
#     pass

# TODO
# update module also
# def test_add_module():
#     pass

# TODO
# def test_device_logging():
#     pass

@pytest.mark.todo
@pytest.mark.quick
def test_load_saves():
    # data = {}
    # assertEqual(trainer.load_saves(data),
    #                  (False, "[load_saves()] Missing params \"weights\""))
    # data = {"weights": "meh"}
    # assertEqual(trainer.load_saves(data),
    #                  (False, "[load_saves()] Invalid or no such method"))
    # data = {"weights": "meh", "method": "load"}
    # assertEqual(trainer.load_saves(data),
    #                  (False, "[load_saves()] No such file"))
    pass
