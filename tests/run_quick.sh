#! /bin/bash

pytest -m 'quick and not todo' test_trainer_training_steps.py && \
pytest -m 'quick and not todo' test_trainer_models.py && \
pytest -m 'quick and not todo' -k 'not test_trainer_models and not test_trainer_training_steps'
