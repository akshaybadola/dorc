python -m pytest -m 'threaded and not todo' -k 'not test_trainer_metrics' && \
python -m pytest -m 'not todo' test_trainer_metrics.py && \
python -m pytest -m 'http and not todo' -k 'not daemon_modules_http' && \
python -m pytest -m 'not todo' test_daemon_modules_http.py
