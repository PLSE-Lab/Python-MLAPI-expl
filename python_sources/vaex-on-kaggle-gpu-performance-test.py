#!/usr/bin/env python
# coding: utf-8

# ### Vaex performance test (GPU instance)
# 
# The purpose of this Notebook is to evaluate the performance of the [vaex](https://github.com/vaexio/vaex) DataFrame library on a Kaggle instance with an active GPU.
# 
# The process is simple:
# - Install vaex
# - Generate a "big-ish" dataset (100_000_000 rows) by replicating the Iris flower dataset
# - Create a couple of computationally expensive virtual columns
# - Evaluate them on the fly via `numpy`, `jit_numba` and `jit_cuda` and compare performance.

# In[ ]:


# Install vaex & pythran from pip
get_ipython().system('pip install vaex')
get_ipython().system('pip install pythran')


# In[ ]:


# Import packages
import vaex
import vaex.ml

import numpy as np

import pylab as plt
import seaborn as sns

from tqdm.notebook import tqdm

import time


# The following method replicates the Iris flower dataset many times, and creates a hdf5 file on disk comprising ~1e8 samples. The purpose is to create "big-ish" data for various types of performance testing. 

# In[ ]:


df = vaex.ml.datasets.load_iris_1e8()


# In[ ]:


# Get a preview of the data
df


# Let us define a simple function that will measure the execution time of other functions (vaex methods).

# In[ ]:


def benchmark(func, reps=5):
    times = []
    for i in tqdm(range(reps), leave=False, desc='Benchmark in progress...'):
        start_time = time.time()
        res = func()
        times.append(time.time() - start_time)
    return np.mean(times), res


# Now let's do some performance testing. I have defined some function, just on top of my head that is a bit computationally challenging to be calculated on the fly. The idea is to see how fast vaex performs when the computations are done with numpy, numba, pythran and cuda. 

# In[ ]:


def some_heavy_function(x1, x2, x3, x4):
    a = x1**2 + np.sin(x2/x1) + (np.tan(x2**2) - x4**(2/3))
    b = (x1/x3)**(0.3) + np.cos(x1) - np.sqrt(x2) - x4**3
    return a**(2/3) / np.tan(b)


# In[ ]:


# Numpy
df['func_numpy'] = some_heavy_function(df.sepal_length, df.sepal_width, 
                                       df.petal_length, df.petal_width)

# Numba
df['func_numba'] = df.func_numpy.jit_numba()

# Pythran
df['func_pythran'] = df.func_numpy.jit_pythran()

# CUDA
df['func_cuda'] = df.func_numpy.jit_cuda()


# In[ ]:


# Calculation of the sum of the virtual columns - this forces their evaluation
duration_numpy, res_numpy =  benchmark(df.func_numpy.sum)
duration_numba, res_numba =  benchmark(df.func_numba.sum)
duration_pythran, res_pythran =  benchmark(df.func_pythran.sum)
duration_cuda, res_cuda =  benchmark(df.func_cuda.sum)


# In[ ]:


print(f'Result from the numpy sum {res_numpy:.5f}')
print(f'Result from the numba sum {res_numba:.5f}')
print(f'Result from the pythran sum {res_pythran:.5f}')
print(f'Result from the cuda sum {res_cuda:.5f}')


# In[ ]:


# Calculate the speed-up compared to the (base) numpy computation
durations = np.array([duration_numpy, duration_numba, duration_pythran, duration_cuda])
speed_up = duration_numpy / durations

# Compute
compute = ['numpy', 'numba', 'pythran', 'cuda']


# In[ ]:


# Let's visualise it

plt.figure(figsize=(16, 6))

plt.subplot(121)
plt.bar(compute, speed_up)
plt.tick_params(labelsize=14)

for i, (comp, speed) in enumerate(zip(compute, speed_up)):
    plt.annotate(s=f'x {speed:.1f}', xy=(i-0.1, speed+0.3), fontsize=14)
plt.annotate(s='(higher is better)', xy=(0, speed+2), fontsize=16)

plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)
plt.xlabel('Accelerators', fontsize=14)
plt.ylabel('Speed-up wrt numpy', fontsize=14)
plt.ylim(0, speed_up[-1]+5)

plt.subplot(122)
plt.bar(compute, durations)
plt.tick_params(labelsize=14)

for i, (comp, duration) in enumerate(zip(compute, durations)):
    plt.annotate(s=f'{duration:.1f}s', xy=(i-0.1, duration+0.3), fontsize=14)
plt.annotate(s='(lower is better)', xy=(2, durations[0]+3), fontsize=16)

plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)
plt.xlabel('Accelerators', fontsize=14)
plt.ylabel('Duration [s]', fontsize=14)
plt.ylim(0, durations[0]+5)


plt.tight_layout()
plt.show()


# Let us try another involved function, this time one calculating the arc-distance between two points on a sphere. We don't have such data here, but let's use this anyway in order to test the speed of the computations.

# In[ ]:


def arc_distance(theta_1, phi_1, theta_2, phi_2):
    temp = (np.sin((theta_2-theta_1)/2*np.pi/180)**2
           + np.cos(theta_1*np.pi/180)*np.cos(theta_2*np.pi/180) * np.sin((phi_2-phi_1)/2*np.pi/180)**2)
    distance = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1-temp))
    return distance * 3958.8


# In[ ]:


# Numpy
df['arc_distance_numpy'] = arc_distance(df.sepal_length, df.sepal_width, 
                                       df.petal_length, df.petal_width)

# Numba
df['arc_distance_numba'] = df.arc_distance_numpy.jit_numba()

# Pythran
df['arc_distance_pythran'] = df.arc_distance_numpy.jit_pythran()

# CUDA
df['arc_distance_cuda'] = df.arc_distance_numpy.jit_cuda()


# In[ ]:


# Calculation of the sum of the virtual columns - this forces their evaluation
duration_numpy, res_numpy =  benchmark(df.arc_distance_numpy.sum)
duration_numba, res_numba =  benchmark(df.arc_distance_numba.sum)
duration_pythran, res_pythran =  benchmark(df.arc_distance_pythran.sum)
duration_cuda, res_cuda =  benchmark(df.arc_distance_cuda.sum)


# In[ ]:


print(f'Result from the numpy sum {res_numpy:.5f}')
print(f'Result from the numba sum {res_numba:.5f}')
print(f'Result from the pythran sum {res_pythran:.5f}')
print(f'Result from the cuda sum {res_cuda:.5f}')


# In[ ]:


# Calculate the speed-up compared to the (base) numpy computation
durations = np.array([duration_numpy, duration_numba, duration_pythran, duration_cuda])
speed_up = duration_numpy / durations


# In[ ]:


# Let's visualise it

plt.figure(figsize=(16, 6))

plt.subplot(121)
plt.bar(compute, speed_up)
plt.tick_params(labelsize=14)

for i, (comp, speed) in enumerate(zip(compute, speed_up)):
    plt.annotate(s=f'x {speed:.1f}', xy=(i-0.1, speed+0.3), fontsize=14)
plt.annotate(s='(higher is better)', xy=(0, speed+2), fontsize=16)

plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)
plt.xlabel('Accelerators', fontsize=14)
plt.ylabel('Speed-up wrt numpy', fontsize=14)
plt.ylim(0, speed_up[-1]+5)

plt.subplot(122)
plt.bar(compute, durations)
plt.tick_params(labelsize=14)

for i, (comp, duration) in enumerate(zip(compute, durations)):
    plt.annotate(s=f'{duration:.1f}s', xy=(i-0.1, duration+0.3), fontsize=14)
plt.annotate(s='(lower is better)', xy=(2, durations[0]+3), fontsize=16)

plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)
plt.xlabel('Accelerators', fontsize=14)
plt.ylabel('Duration [s]', fontsize=14)
plt.ylim(0, durations[0]+5)


plt.tight_layout()
plt.show()

