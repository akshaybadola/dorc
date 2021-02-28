#! /bin/bash

python -m pytest -m 'not todo' test_trainer_models.py test_trainer_device.py
