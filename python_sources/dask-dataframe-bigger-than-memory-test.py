#!/usr/bin/env python
# coding: utf-8

# This is a CPU only kernel simply to test dask dataframe's capability to handle dataframes larger than cpu memory.
# 
# Takeaways:
# * dask can handle dataframes larger than memory by breaking it down into chunks.
# * dask arrary operations are utilizing multi-threads out of the box.
# * element wise operation such as masking, reduction can be done in reasonable time.
# * groupby-aggregation might be doable but it is too slow to be useful.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
import dask
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls -lsh ../input/PLAsTiCC-2018/test_set.csv')


# In[ ]:


# monitor cpu memory usage
get_ipython().system('free -g')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# kernel died running the following pandas command due to OOM\n#df = pd.read_csv('../input/PLAsTiCC-2018/test_set.csv') \n\ndf = dd.read_csv('../input/PLAsTiCC-2018/test_set.csv') ")


# dask is lazy so the dataframe is NOT read yet but we can still access the header instantly

# In[ ]:


df


# As shown above, dask breaks down the big dataframe into 310 chunks.

# In[ ]:


df.head() # dask just reads the head


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df.shape # dask is not getting the actual shape since it is lazy')


# The number of rows is a `delayed` object and the number of columns is trivial to get, which is `6`

# In[ ]:


get_ipython().run_cell_magic('time', '', 'dask.compute(df.shape)')


# It should be noted that the wall time is roughly 25% of the total CPU time, indicating that `dask` is using 4 threads to do things in parallel.

# Let's caculate the mean value of a column.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# simple column-wise reduction operations\ndf['flux'].mean().compute() # returns a scalar")


# By using the `compute` method, dask reads the big csv file chunk by chunk and calculate the mean value. 

# In[ ]:


get_ipython().system('free -g')


# Notice that the CPU memory usage is not increased since dask has already released memory of all the intermediate variables in the process of calculating `mean`

# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_sample = df.loc[df.object_id==13].compute()')


# Next let's do a grouby aggregation

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# it runs for more than 9 hours and is killed by kaggle.\n#flux_stats_of_each_mjd = df.groupby('mjd').agg({'flux':['std']}).compute()\n# This will return a pandas dataframe\n\n#flux_stats_of_each_mjd.head()")


# In[ ]:


#print(type(flux_stats_of_each_mjd),flux_stats_of_each_mjd.shape)


# `flux_stats_of_each_mjd` is a pandas dataframe. Unfortunately, it is a slow operation that runs for more than 9 hours and killed by kaggle but in theory eventually it should get things done.

# You might think that we need to the whole dataframe into memory to do the groupby aggregation. However, dask adopts an [apply-concat-apply](https://blog.dask.org/2019/10/08/df-groupby) paradigm, where aggregation is done first for each chunk, and then the aggregated intermediate results are concatenated to form a new dataframe, and aggregated again. 
