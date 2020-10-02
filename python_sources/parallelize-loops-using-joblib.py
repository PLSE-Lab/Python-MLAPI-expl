#!/usr/bin/env python
# coding: utf-8

# # Parallelize embarrassing  for loops using [Joblib](joblib.readthedocs.io/)

# In[ ]:


get_ipython().system('pip install joblib')
get_ipython().system('pip install Pillow')


# ### Import the `Parallel` class

# In[ ]:


from joblib import Parallel, delayed


# ### Import other modules

# In[ ]:


from PIL import ImageDraw, Image
import numpy as np
from pathlib import Path
from time import sleep, time
from multiprocessing import cpu_count


# ## Let's start with a simple example

# In[ ]:


size_w = size_h = 512


# ### Define a function

# In[ ]:


def draw_rectangles(img_index, save_dir, n, m):
    image = Image.new(mode = 'RGB', size = (n, m), color = (255, 255, 255))
    draw = ImageDraw.Draw(image)
    sleep(3.0)
    x1 = np.random.randint(low=0, high=n//2)
    x2 = np.random.randint(low=n//2 + 1, high=n)
    
    y1 = np.random.randint(low=0, high=m//2)
    y2 = np.random.randint(low=m//2 + 1, high=m)
    
    draw.rectangle(xy=[(x1,y1), (x2,y2)], outline=(255, 0, 0))
    image_name = img_index + '.png'
    image.save(save_dir.joinpath(image_name).as_posix())
    return image_name


# ### Without Parallel processing

# In[ ]:


save_dir_no_parallel_process = Path('./no_parallel_process')
save_dir_no_parallel_process.mkdir(parents=True, exist_ok=True)


# In[ ]:


start_time = time()


for image_index in range(10):
    image_name = draw_rectangles(img_index=str(image_index+1), save_dir=save_dir_no_parallel_process, n=size_w, m=size_h)
    print("Image Name: ", image_name)


sequential_execution_time = time() - start_time


print("Execution Time: ", sequential_execution_time)


# ### With Joblib `Parallel`

# In[ ]:


save_dir_parallel_process = Path('./parallel_process')
save_dir_parallel_process.mkdir(parents=True, exist_ok=True)


# In[ ]:


start_time = time()


print("Number of jobs: ",int(cpu_count()))

# Use multiple CPUs (Multi Processing)
image_filenames = Parallel(n_jobs=int(cpu_count()), prefer='processes')(
    delayed(draw_rectangles)(img_index=str(image_index+1), save_dir=save_dir_parallel_process, n=size_w, m=size_h) 
    for image_index in range(10)
)

parallel_execution_time = time() - start_time


print("Execution Time: ", parallel_execution_time)


# <span style="color:red">Note the time difference in the cell execution time of parallel vs non-parallel</span>

# #### Let's print the result

# In[ ]:


for img_index in image_filenames:
    print(img_index)


# ## We can do a lot with Joblib `Parallel`:
# #### 1. [Thread-based parallelism vs process-based parallelism](https://joblib.readthedocs.io/en/latest/parallel.html#thread-based-parallelism-vs-process-based-parallelism)
# #### 2. [Shared-memory](https://joblib.readthedocs.io/en/latest/parallel.html#shared-memory-semantics)
# #### 3. [Working with numerical data in shared memory (memmapping)](https://joblib.readthedocs.io/en/latest/parallel.html#working-with-numerical-data-in-shared-memory-memmapping)

# ### In this tutorial I will be discussing the first two points, as these are the ones we encounter the most in daily routine work

# ## 1. Choose backend

# ### Parallelism can be achieved in two ways: multi-threading and multi-processing
# 
# #### In Joblib, we can specify the backend type and backend:
# 
# <img src="https://raw.githubusercontent.com/karanpathak/blog/master/joblib/imgs/joblib_backends.png" alt="Joblib backend and backend type" width="500" height="500"/>
# 
# > By default, Joblib `Parallel` uses `loky` backend
# 
# **<u>P.S.</u>** - We can also use dask backend for parallelizing.

# #### [Here](#With-Joblib-Parallel-processing), I had already showing an example of multi-processing with `loky` backend
# ### Let's choose `threading` backend with argument `prefer='threads'`

# In[ ]:


save_dir_parallel_threads = Path('./parallel_threads')
save_dir_parallel_threads.mkdir(parents=True, exist_ok=True)


# In[ ]:


start_time = time()


print("Number of threads: ",10)

# Use multiple CPUs (Multi Processing)
image_filenames = Parallel(prefer='threads', n_jobs=10)(
    delayed(draw_rectangles)(img_index=str(image_index+1), save_dir=save_dir_parallel_threads, n=size_w, m=size_h) 
    for image_index in range(10)
)

parallel_execution_time_threading = time() - start_time


print("Execution Time: ", parallel_execution_time_threading)


# <br>

# ## 2. Updating shared memory in parallel processing 

# #### if the parallel function needs to rely on the shared memory semantics of threads, it should be made explicit with `require='sharedmem'`

# ### Let's see an example

# #### Shared object

# In[ ]:


shared_list = []


# #### Function to update the shared object

# In[ ]:


def add_to_list(x):
    sleep(3.0)
    shared_list.append(x)


# #### Update the shared object parallelly

# In[ ]:


start_time = time()

result = Parallel(n_jobs=cpu_count(), require='sharedmem')(delayed(add_to_list)(i) for i in range(10))

print("Execution Time: ", time()-start_time)


# #### Let's check if our shared object has been updated

# In[ ]:


shared_list


# <br>

# # Summary

# This tutorial showcases, how to use [Joblib](joblib.readthedocs.io) to parallelize `loops` without using any other heavy modules like apache spark etc which usually have a creational overhead.
# 
# - [Part 1](#Lets-start-with-a-simple-example) - Shows a simple example on how to parallelize `for loop` using `Parallel` class. It highlights the difference in execution time for parallel vs squential approach.
# - [Part 2](#1.-Choose-backend) - Shows different available backends (multi-processing and multi-threading) in Joblib and how to choose a particluar backend.
# - [Part 3](#2.-Updating-shared-memory-in-parallel-processing) - Shows how we can interact with shared memory objects.
# 

# ## Looking forward for your feedback in the comments section below
# ### If you liked this kernel please hit the Upvote button.

# ## Next: Try Disk caching and save lot of computation time using Joblib: https://www.kaggle.com/karanpathak/disk-caching-using-joblib

# In[ ]:




