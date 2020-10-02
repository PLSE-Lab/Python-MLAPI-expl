#!/usr/bin/env python
# coding: utf-8

# <a id="introduction"></a>
# ## Introduction to Dask
# #### by Paul Hendricks
# #### modified by Beniel Thileepan 
# -------
# 
# This work is modified inorder to run in Kaggle with additional rapids dataset.integers and strings.
# 
# In this notebook, we will show how to work with cuDF DataFrames in RAPIDS (https://www.kaggle.com/cdeotte/rapids)
# 
# **Table of Contents**
# 
# * [Introduction to Dask](#introduction)
# * [Setup](#setup)
# * [Introduction to Dask](#dask)
# * [Conclusion](#conclusion)

# <a id="setup"></a>
# ## Setup and install RAPIDs
# 

# In[ ]:


get_ipython().system('nvidia-smi')


# Next, let's see what CUDA version we have:

# In[ ]:


get_ipython().system('nvcc --version')


# In[ ]:


import sys
get_ipython().system('rsync -ah --progress ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('rsync -ah --progress /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import cudf


# ## Install graphviz
# The visualizations in this notebook require graphviz.  Your environment may not have it installed, but don't worry! If you don't, we're going to install it now.  This can take a little while, so sit tight.

# In[ ]:


import os
try:
    import graphviz
except ModuleNotFoundError:
    os.system('conda install -c conda-forge graphviz -y')
    os.system('conda install -c conda-forge python-graphviz -y')


# <a id="dask"></a>
# ## Introduction to Dask
# 
# Dask is a library the allows for parallelized computing. Written in Python, it allows one to compose complex workflows using large data structures like those found in NumPy, Pandas, and cuDF. In the following examples and notebooks, we'll show how to use Dask with cuDF to accelerate common ETL tasks as well as build and train machine learning models like Linear Regression and XGBoost.
# 
# To learn more about Dask, check out the documentation here: http://docs.dask.org/en/latest/
# 
# #### Client/Workers
# 
# Dask operates by creating a cluster composed of a "client" and multiple "workers". The client is responsible for scheduling work; the workers are responsible for actually executing that work. 
# 
# Typically, we set the number of workers to be equal to the number of computing resources we have available to us. For CPU based workflows, this might be the number of cores or threads on that particlular machine. For example, we might set `n_workers = 8` if we have 8 CPU cores or threads on our machine that can each operate in parallel. This allows us to take advantage of all of our computing resources and enjoy the most benefits from parallelization.
# 
# On a system with one or more GPUs, we usually set the number of workers equal to the number of GPUs available to us. Dask is a first class citizen in the world of General Purpose GPU computing and the RAPIDS ecosystem makes it very easy to use Dask with cuDF and XGBoost. 
# 
# Before we get started with Dask, we need to setup a Local Cluster of workers to execute our work and a Client to coordinate and schedule work for that cluster. As we see below, we can inititate a `cluster` and `client` using only few lines of code.

# In[ ]:


import dask; print('Dask Version:', dask.__version__)
from dask.distributed import Client, LocalCluster


# create a local cluster with 4 workers
n_workers = 4
cluster = LocalCluster(n_workers=n_workers)
client = Client(cluster)


# Let's inspect the `client` object to view our current Dask status. We should see the IP Address for our Scheduler as well as the the number of workers in our Cluster. 

# In[ ]:


# show current Dask status
client


# You can also see the status and more information at the Dashboard, found at `http://<ip_address>/status`. This can be ignored now since this is pointing to local machine.
# 
# With our client and workers setup, it's time to execute our first program in parallel. We'll define a function called `add_5_to_x` that takes some value `x` and adds 5 to it.

# In[ ]:


def add_5_to_x(x):
    return x + 5


# Next, we'll iterate through our `n_workers` and create an execution graph, where each worker is responsible for taking its ID and passing it to the function `add_5_to_x`. For example, the worker with ID 2 will take its ID and pass it to the function `add_5_to_x`, resulting in the value 7.

# In[ ]:


from dask import delayed


addition_operations = [delayed(add_5_to_x)(i) for i in range(n_workers)]
addition_operations


# The above output shows a list of several `Delayed` objects. An important thing to note is that the workers aren't actually executing these results - we're just defining the execution graph for our client to execute later. The `delayed` function wraps our function `add_5_to_x` and returns a `Delayed` object. This ensures that this computation is in fact "delayed" - or lazily evaluated - and not executed on the spot i.e. when we define it.
# 
# Next, let's sum each one of these intermediate results. We can accomplish this by wrapping Python's built-in `sum` function using our `delayed` function and storing this in a variable called `total`.

# In[ ]:


total = delayed(sum)(addition_operations)
total


# Using the `graphviz` library, we can use the `visualize` method of a `Delayed` object to visualize our current graph.

# In[ ]:


total.visualize()


# As we mentioned before, none of these results - intermediate or final - have actually been compute. We can compute them using the `compute` method of our `client`.

# In[ ]:


from dask.distributed import wait
import time


addition_futures = client.compute(addition_operations, optimize_graph=False, fifo_timeout="0ms")
total_future = client.compute(total, optimize_graph=False, fifo_timeout="0ms")
wait(total_future)  # this will give Dask time to execute the work


# Let's inspect the output of each call to `client.compute`:

# In[ ]:


addition_futures


# We can see from the above output that our `addition_futures` variable is a list of `Future` objects - not the "actual results" of adding 5 to each of `[0, 1, 2, 3]`. These `Future` objects are a promise that at one point a computation will take place and we will be left with a result. Dask is responsible for fulfilling that promise by delegating that task to the appropriate Dask worker and collecting the result.
# 
# Let's take a look at our `total_future` object:

# In[ ]:


print(total_future)
print(type(total_future))


# Again, we see that this is an object of type `Future` as well as metadata about the status of the request (i.e. whether it has finished or not), the type of the result, and a key associated with that operation. To collect and print the result of each of these `Future` objects, we can call the `result()` method.

# In[ ]:


addition_results = [future.result() for future in addition_futures]
print('Addition Results:', addition_results)


# Now we see the results that we want from our addition operations. We can also use the simpler syntax of the `client.gather` method to collect our results.

# In[ ]:


addition_results = client.gather(addition_futures)
total_result = client.gather(total_future)
print('Addition Results:', addition_results)
print('Total Result:', total_result)


# Awesome! We just wrote our first distributed workflow.
# 
# To confirm that Dask is truly executing in parallel, let's define a function that sleeps for 1 second and returns the string "Success!". In serial, this function should take our 4 workers around 4 seconds to execute.

# In[ ]:


def sleep_1():
    time.sleep(1)
    return 'Success!'


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfor _ in range(n_workers):\n    sleep_1()')


# As expected, our process takes about 4 seconds to run. Now let's execute this same workflow in parallel using Dask.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# define delayed execution graph\nsleep_operations = [delayed(sleep_1)() for _ in range(n_workers)]\n\n# use client to perform computations using execution graph\nsleep_futures = client.compute(sleep_operations, optimize_graph=False, fifo_timeout="0ms")\n\n# collect and print results\nsleep_results = client.gather(sleep_futures)\nprint(sleep_results)')


# Using Dask, we see that this whole process takes a little over a second - each worker is executing in parallel!

# ## Dask Cudf

# Let's start by creating a local cluster of workers and a client to interact with that cluster.

# In[ ]:


from dask.distributed import Client
from dask_cuda import LocalCUDACluster


# create a local CUDA cluster
cluster = LocalCUDACluster()
client = Client(cluster)
client


# We'll define a function called load_data that will create a cudf.DataFrame with two columns, key and value. The column key will be randomly filled with either a 0 or a 1, with 50% probability of either number being selected. The column value will be randomly filled with numbers sampled from a normal distribution.

# In[ ]:


import cudf; print('cuDF Version:', cudf.__version__)
import numpy as np; print('NumPy Version:', np.__version__)


def load_data(n_rows):
    df = cudf.DataFrame()
    random_state = np.random.RandomState(43210)
    df['key'] = random_state.binomial(n=1, p=0.5, size=(n_rows,))
    df['value'] = random_state.normal(size=(n_rows,))
    return df


# We'll also define a function head that takes a cudf.DataFrame and returns the first 5 rows.

# In[ ]:


def head(dataframe):
    return dataframe.head()


# We'll define the number of workers as well as the number of rows each dataframe will have.

# In[ ]:


# define the number of workers
n_workers = 1  # feel free to change this depending on how many GPUs you have

# define the number of rows each dataframe will have
n_rows = 125000000  # we'll use 125 million rows in each dataframe


# We'll create each dataframe using the delayed operator.

# In[ ]:


from dask.delayed import delayed


# create each dataframe using a delayed operation
dfs = [delayed(load_data)(n_rows) for i in range(n_workers)]
dfs


# We see the result of this operation is a list of Delayed objects. It's important to note that these operations are "delayed" - nothing has been computed yet, meaning our data has not yet been created!
# 
# We can apply the head function to each of our "delayed" dataframes.

# In[ ]:


head_dfs = [delayed(head)(df) for df in dfs]
head_dfs


# 
# 
# As before, we see that the result is a list of Delayed objects - an important thing to note is that our "key", or unique identifier for each operation, has changed. You should see the name of the function head followed by a hash sign. For example, one might see:
# 
#  [Delayed('head-8e946db2-feaf-4e79-99ab-f732b6e28461')
#  Delayed('head-eb06bc77-9d5c-4a47-8c01-b5b36710b727')]
# 
# Again, nothing has been computed - let's compute the results and execute the workflow using the client.compute() method.
# 

# In[ ]:


from dask.distributed import wait


# use the client to compute - this means create each dataframe and take the head
futures = client.compute(head_dfs)
wait(futures)  # this will give Dask time to execute the work before moving to any subsequently defined operations
futures


# 
# 
# We see that our results are a list of futures. Each object in this list tells us a bit information about itself: the status (pending, error, finished), the type of the object, and the key (unique identifief).
# 
# We can use the client.gather method to collect the results of each of these futures.
# 

# In[ ]:


# collect the results
results = client.gather(futures)
results


# We see that our results are a list of cuDF DataFrames, each having 2 columns and 5 rows. Let's inspect the first dataframe:

# In[ ]:


# let's inspect the head of the first dataframe
print(results[0])


# That was a pretty simple example. Let's see how we can use this perform a more complex operation like figuring how many total rows we have across all of our dataframes. We'll define a function called length that will take a cudf.DataFrame and return the first value of the shape attribute i.e. the number of rows for that particular dataframe.

# In[ ]:


def length(dataframe):
    return dataframe.shape[0]


# We'll define our operation on the dataframes we've created:

# In[ ]:


lengths = [delayed(length)(df) for df in dfs]
lengths


# And then use Python's built-in sum function to sum all of these lengths.

# In[ ]:


total_number_of_rows = delayed(sum)(lengths)


# At this point, total_number_of_rows hasn't been computed yet. But we can still visualize the graph of operations we've defined using the visualize() method.

# In[ ]:


total_number_of_rows.visualize()


# The graph can be read from bottom to top. We see that for each worker, we will first execute the load_data function to create each dataframe. Then the function length will be applied to each dataframe; the results from these operations on each worker will then be combined into a single result via the sum function.
# 
# Let's now execute our workflow and compute a value for the total_number_of_rows variable.

# In[ ]:


# use the client to compute the result and wait for it to finish
future = client.compute(total_number_of_rows)
wait(future)
future


# We see that our computation has finished - our result is of type int. We can collect our result using the client.gather() method.

# In[ ]:


# collect result
result = client.gather(future)
result


# That's all there is to it! We can define even more complex operations and workflows using cuDF DataFrames by using the delayed, wait, client.submit(), and client.gather() workflow.
# 
# However, there can sometimes be a drawback from using this pattern. For example, consider a common operation such as a groupby - we might want to group on certain keys and aggregate the values to compute a mean, variance, or even more complex aggregations. Each dataframe is located on a different GPU - and we're not guaranteed that all of the keys necessary for that groupby operation are located on a single GPU i.e. keys may be scattered across multiple GPUs.
# 
# To make our problem even more concrete, let's consider the simple operation of grouping on our key column and calculating the mean of the value column. To sovle this problem, we'd have to sort the data and transfer keys and their associated values from one GPU to another - a tricky thing to do using the delayed pattern. In the example below, we'll show an example of this issue with the delayed pattern and motivate why one might consider using the dask_cudf API.
# 
# First, let's define a function groupby that takes a cudf.DataFrame, groups by the key column, and calculates the mean of the value column.

# In[ ]:


def groupby(dataframe):
    return dataframe.groupby('key')['value'].mean()


# We'll apply the function groupby to each dataframe using the delayed operation.

# In[ ]:


groupbys = [delayed(groupby)(df) for df in dfs]


# We'll then execute that operation:

# In[ ]:


# use the client to compute the result and wait for it to finish
groupby_dfs = client.compute(groupbys)
wait(groupby_dfs)
groupby_dfs


# In[ ]:


results = client.gather(groupby_dfs)
results


# In[ ]:


for i, result in enumerate(results):
    print('cuDF DataFrame:', i)
    print(result)


# This isn't exactly what we wanted though - ideally, we'd get one dataframe where for each unique key (0 and 1), we get the mean of the value column.
# 
# We can use the dask_cudf API to help up solve this problem. First we'll import the dask_cudf library and then use the dask_cudf.from_delayed function to convert our list of delayed dataframes to an object of type dask_cudf.core.DataFrame. We'll use this object - distributed_df - along with the dask_cudf API to perform that "tricky" groupby operation.

# In[ ]:


import dask_cudf; print('Dask cuDF Version:', dask_cudf.__version__)


# create a distributed cuDF DataFrame using Dask
distributed_df = dask_cudf.from_delayed(dfs)
print('Type:', type(distributed_df))
distributed_df


# The dask_cudf API closely mirrors the cuDF API. We can use a groupby similar to how we would with cuDF - but this time, our operation is distributed across multiple GPUs!

# In[ ]:


result = distributed_df.groupby('key')['value'].mean().compute()
result


# Lastly, let's examine our result!

# In[ ]:


print(result)


# <a id="conclusion"></a>
# ## Conclusion
# 
# In this tutorial, we learned how to use Dask with basic Python primitives like integers and strings.
# 
# To learn more about RAPIDS, be sure to check out: 
# 
# * [Open Source Website](http://rapids.ai)
# * [GitHub](https://github.com/rapidsai/)
# * [Press Release](https://nvidianews.nvidia.com/news/nvidia-introduces-rapids-open-source-gpu-acceleration-platform-for-large-scale-data-analytics-and-machine-learning)
# * [NVIDIA Blog](https://blogs.nvidia.com/blog/2018/10/10/rapids-data-science-open-source-community/)
# * [Developer Blog](https://devblogs.nvidia.com/gpu-accelerated-analytics-rapids/)
# * [NVIDIA Data Science Webpage](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/)
