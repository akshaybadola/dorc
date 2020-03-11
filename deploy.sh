#! /bin/bash

bleh=${@}
if (( ${#@} > 0))
then if [ ${bleh[0]} = "trainer" ]
     then
	 scp build/trainer.cpython-37m-x86_64-linux-gnu.so mc15pc15@10.5.0.96:~/trainer/
	 scp trainer/autoloads.py mc15pc15@10.5.0.96:~/trainer/
     elif [ ${bleh[0]} = "both" ]
     then
     	 scp build/trainer.cpython-37m-x86_64-linux-gnu.so mc15pc15@10.5.0.96:~/trainer/
	 scp trainer/autoloads.py mc15pc15@10.5.0.96:~/trainer/
	 scp dist/* mc15pc15@10.5.0.96:~/trainer/dist/
     elif [ ${bleh[0]} = "dist" ]
     then scp dist/* mc15pc15@10.5.0.96:~/trainer/dist/
     else
	 echo "Strange option given"
     fi
else
    scp build/trainer.cpython-37m-x86_64-linux-gnu.so mc15pc15@10.5.0.96:~/trainer/
    scp trainer/autoloads.py mc15pc15@10.5.0.96:~/trainer/
    scp dist/* mc15pc15@10.5.0.96:~/trainer/dist/
fi



