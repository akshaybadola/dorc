#! /bin/bash

bash quick_test.sh
bash threaded_test.sh
bash http_test.sh
python -m pytest -m 'quick and not todo' test_trainer_device.py && \
python -m pytest -m 'quick and not todo' test_trainer_models.py
