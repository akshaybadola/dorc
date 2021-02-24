#! /bin/bash

bash quick_test.sh
python -m pytest -m 'quick and not todo' test_trainer_load_save.py && \
python -m pytest -m 'quick and not todo' test_trainer_device.py && \
python -m pytest -m 'quick and not todo' test_trainer_models.py
