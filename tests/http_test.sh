#! /bin/bash

python -m pytest -m 'http and not todo' -k 'not daemon_modules_http' && \
python -m pytest -m 'not todo' test_daemon_modules_http.py
