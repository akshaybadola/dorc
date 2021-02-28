#! /bin/bash

pytest --collect-only -m 'not spec and not quick and not http and not threaded and not todo and not gpus and not ddp' $@
