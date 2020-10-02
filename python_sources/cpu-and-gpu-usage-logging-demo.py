#!/usr/bin/env python
# coding: utf-8

# # CPU and GPU logging demo
# This kernel uses my script from https://github.com/scottclowe/cpu-gpu-utilisation-logging-python to track the CPU and GPU utilisation while the kernel is running.
# 
# In this demo kernel, we have the network do matrix multiplication repeatedly in a simulation of useful computation.

# ## Downloading and running the logging script
# We start a background process running which, at regular intervals, records the CPU and GPU compute and memory utilisation to a file. We do this as a background process because that way it can record everything that happens while the notebook is running, and it doesn't block the notebook from moving on from the cell to the next cell.
# 
# Starting a subprocess with Popen gives us a process ID, which we need to close when we've finished with the logging (see the final cell in this notebook).
# 
# As a cautionary note, I have obsereved that the background process performing the logging is sometimes killed prematurely. If your code is running something which prints output, you will see an extra blank line in the output whenever this occurs (both things occur because of something going on periodically behind the scenes on the server, I'm not sure what).

# In[ ]:


# Download the logging script
get_ipython().system('wget https://raw.githubusercontent.com/scottclowe/cpu-gpu-utilisation-logging-python/master/log_gpu_cpu_stats.py')


# In[ ]:


# Start the logger running in a background process. It will keep running until you tell it to stop.
# We will save the CPU and GPU utilisation stats to a CSV file every 0.2 seconds.
import subprocess
get_ipython().system('rm -f log_compute.csv')
logger_fname = 'log_compute.csv'
logger_pid = subprocess.Popen(
    ['python', 'log_gpu_cpu_stats.py',
     logger_fname,
     '--loop',  '0.2',  # Interval between measurements, in seconds (optional, default=1)
    ])
print('Started logging compute utilisation')


# ## Running simulated computation on CPU and GPU
# We're going to do adds, multiplies, and matrix multiplies with matrices of different sizes. As the size increases, the number of operations to be simultaneously performed increases. We can see that the CPU is faster with small matrices, but the GPU has so many threads available to it that there is virtually no penalty for increasing the size of the matrix up to a certain point. If you change the option over to random floats instead of ones, you'll see a big slow down due to the increase in time generating the matrices. I haven't investigated half/double precision here because Pytorch doesn't support it on CPU.
# 
# Because there is a large disparity in the rate at which iterations can be performed, we fix the duration of compute and see how many iterations could be achieved in that timespan.

# In[ ]:


import os
import time

import numpy as np
import torch


# In[ ]:


t_per_exp = 2
t_sleep = 2

tgen_list = [
    ('ones_float', lambda s, d: torch.ones(s, dtype=torch.float, device=d)),
    #('rand_float', lambda s, d: torch.rand(s, dtype=torch.float, device=d)),
]
op_list = [
    ('ADD', lambda x, y: x + y),
    ('MUL', lambda x, y: x * y),
    ('MATMUL', lambda x, y: torch.matmul(x, y)),
]

# Do some compute on CPU and GPU
for regen_tensors in [False, True]:
    for tgen_name, tgen_fn in tgen_list:
        print("\n{} tensors ({})...".format(tgen_name, 'regenerate inputs' if regen_tensors else 'static inputs'))
        for op_name, op_fun in op_list:
            print("\n  {} operations...".format(op_name))
            time.sleep(5)
            for device in ['cpu', 'cuda']:
                for shp in [(8, 8), (64, 64), (512, 512), (4096, 4096)]: #[(10, 10), (100, 100), (1000, 1000), (10000, 10000)]:
                    print(
                        '    Beginning {:<12} {} {:<6} operations on {:<4} for {}s ({})'
                        ''.format(str(shp), tgen_name, op_name, device.upper(), t_per_exp,
                                 'regenerate inputs' if regen_tensors else 'static inputs')
                    )
                    i = 0
                    t_start = time.time()
                    t_gen = 0
                    t_op = 0
                    if not regen_tensors:
                        x = tgen_fn(shp, device)
                        y = tgen_fn(shp, device)
                    while time.time() - t_start < t_per_exp:
                        t0 = time.time()
                        if regen_tensors:
                            x = tgen_fn(shp, device)
                            y = tgen_fn(shp, device)
                        t1 = time.time()
                        t_gen += t1 - t0
                        z = op_fun(x, y)
                        t_op += time.time() - t1
                        i += 1
                    dur = time.time() - t_start
                    print(
                        '      Completed {:>7} iterations in {:.1f}s ({:10.3f}it/s);'
                        ' {:.1f}% was generating tensors'
                        ''.format(i, dur, i / dur, 100 * t_gen / (t_gen + t_op + 0.001))
                    )
                    time.sleep(t_sleep)


# We're done with the simulated work, now lets see what was recorded in the log file.

# In[ ]:


get_ipython().system('head log_compute.csv')


# In[ ]:


get_ipython().system('tail log_compute.csv')


# The GPU is still registered as running at 100%. This happens consistently for me after running at 100% for a while, though I am not sure why the GPU does this. If you wait long enough, it will return to 0%, or if you do a small amount of compute it appears become accurate again.
# 
# Let's wait and see if the GPU level falls back down again.

# In[ ]:


time.sleep(60)
get_ipython().system('tail log_compute.csv')


# ## Plotting utilisation over time
# We load up the logged data from the CSV file and plot the utilisation over time.

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt


# In[ ]:


logger_df = pd.read_csv(logger_fname)


# In[ ]:


logger_df


# We can plot all the utilisation stats at once on a single plot.

# In[ ]:


t = pd.to_datetime(logger_df['Timestamp (s)'], unit='s')
cols = [col for col in logger_df.columns
        if 'time' not in col.lower() and 'temp' not in col.lower()]
plt.figure(figsize=(15, 9))
plt.plot(t, logger_df[cols])
plt.legend(cols)
plt.xlabel('Time')
plt.ylabel('Utilisation (%)')
plt.show()


# Note that the CPU is working hard even when we are running the computation on the GPU!
# 
# We can also plot the graphs individually. Here, we also include the temperature of the GPU (which is in degrees Celcius, not %).

# In[ ]:


for col in logger_df.columns:
    if 'time' in col.lower(): continue
    plt.figure(figsize=(15, 9))
    plt.plot(t, logger_df[col])
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.show()


# ## Finally, close the logging process

# In[ ]:


# End the background process logging the CPU and GPU utilisation.
logger_pid.terminate()
print('Terminated the compute utilisation logger background process')

