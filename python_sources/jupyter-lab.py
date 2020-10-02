txt = """
#!/bin/bash

if [ "$1" = "DAEMON" ]; then
    # is this necessary? Add other signals at will (TTIN TTOU INT STOP TSTP)
    trap '' INT
    cd /tmp
    shift
    ### daemonized section ######
    /tmp/ngrok http 8888
    #### end of daemonized section ####
    exit 0
fi

export PATH=/sbin:/usr/sbin:/bin:/usr/bin:/usr/local/sbin:/usr/local/bin
umask 022
# You can add nice and ionice before nohup but they might not be installed
nohup setsid $0 DAEMON $* 2>/var/log/mydaemon.err >/var/log/mydaemon.log &

"""
open("/daemon.sh", "w").write(txt)


import os

os.system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip  > /dev/null 2>&1')
os.system('unzip ngrok-stable-linux-amd64.zip  > /dev/null 2>&1')
os.system('rm -rf ngrok-stable-linux-amd64.zip  > /dev/null 2>&1')
os.system('mv ./ngrok /tmp/ngrok')
os.system('apt install -q -y cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0 > /dev/null 2>&1')

import subprocess
os.system('chmod +x /daemon.sh')
os.system('sh /daemon.sh')

os.system(""" 
conda install nodejs -y -q > /dev/null 2>&1; 
pip install --upgrade pip  > /dev/null 2>&1; 
pip install torch jupyterlab torchvision matplotlib jupyter-tensorboard tensorflow-gpu jupyter jupyter_contrib_nbextensions jupyter_http_over_ws tensorboard==1.11  > /dev/null 2>&1; 
jupyter-contrib-nbextension install  > /dev/null 2>&1;  
jupyter serverextension enable --py jupyter_http_over_ws  > /dev/null 2>&1 ; 
jupyter serverextension enable --py jupyterlab --sys-prefix  > /dev/null 2>&1 ; 
jupyter labextension install jupyterlab_tensorboard  > /dev/null 2>&1 ; 
jupyter labextension install jupyterlab_vim  > /dev/null 2>&1 ; 
jupyter labextension enable jupyterlab_tensorboard jupyterlab_vim  > /dev/null 2>&1 ;
curl -s http://localhost:4040/api/tunnels | python3 -c   "import sys, json; print('Jupyter lab is running at: {}'.format(json.load(sys.stdin)['tunnels'][0]['public_url']))";
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 jupyter lab --ip=127.0.0.1 --no-browser --allow-root --NotebookApp.allow_origin='https://colab.research.google.com' --NotebookApp.token=''  --NotebookApp.disable_check_xsrf=True & """)

    
import time
time.sleep(60*60*8.8) # 9 hours - 12 mins