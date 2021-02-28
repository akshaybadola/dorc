#! /bin/bash

python -m pytest -m 'quick and not spec and not todo' -k 'not test_trainer_models and not test_trainer_device'
