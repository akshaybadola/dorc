#! /bin/bash

pytest --cov=../dorc --cov-report=html -m 'quick and not spec and not todo' \
       -k 'not test_trainer_models and not test_trainer_device'
echo "Finished quick tests"
pytest --cov=../dorc --cov-report=html --cov-append -m 'threaded and not todo'
echo "Finished threaded tests"
pytest --cov=../dorc --cov-report=html --cov-append -m 'http and not todo'
echo "Finished http tests"
if ${#@}
   then
       pytest --cov=../dorc --cov-report=html --cov-append -m 'not todo' \
              test_trainer_models.py test_trainer_device.py
       echo "Finished devices tests"
fi
pytest --cov=../dorc --cov-report=html --cov-append -m 'spec and not todo and not bug'
echo "Finished all tests"
