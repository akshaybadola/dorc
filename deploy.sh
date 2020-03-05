#! /bin/bash

scp build/trainer.cpython-37m-x86_64-linux-gnu.so mc15pc15@10.5.0.96:~/trainer/
if (( ${#@} > 0))
then scp dist/* mc15pc15@10.5.0.96:~/trainer/dist/
fi



