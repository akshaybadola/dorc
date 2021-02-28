#! /bin/bash

pytest --cov=../dorc --cov-append -m 'quick and not spec and not todo' \
       -k 'not test_trainer_models and not test_trainer_device'
pytest --cov=../dorc --cov-append -m 'threaded and not todo'
pytest --cov=../dorc --cov-append -m 'http and not todo'
pytest --cov=../dorc --cov-append -m 'not todo' test_trainer_models.py test_trainer_device.py
pytest --cov=../dorc --cov-append -m 'spec and not todo and not bug'

