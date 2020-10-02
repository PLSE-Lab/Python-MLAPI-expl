#!/usr/bin/env python
# coding: utf-8

# # Vaex

# ### Vaex, what's that?
# - Vaex is a high performance Python library for lazy Out-of-Core DataFrames (similar to Pandas), to visualize and explore big tabular datasets. 
# - It calculates statistics such as mean, sum, count, standard deviation etc, on an N-dimensional grid for more than a billion (10^9) samples/rows per second. 
# - Visualization is done using histograms, density plots and 3d volume rendering, allowing interactive exploration of big data. Vaex uses memory mapping, zero memory copy policy and lazy computations for best performance (no memory wasted).

# In[ ]:


get_ipython().system('pip install vaex==2.5.0 ')


# In[ ]:


import vaex

import pandas as pd
import numpy as np


# In[ ]:


n_rows = 100000 # one hundred thousand random data
n_cols = 10
df = pd.DataFrame(np.random.randint(0, 100, size=(n_rows, n_cols)), columns=['c%d' % i for i in range(n_cols)])
df.head()


# In[ ]:


df.info(memory_usage='deep')


# ### Creating Csv files

# In[ ]:


file_path = 'main_dataset.csv'
df.to_csv(file_path, index=False)


# ### Create Hdf5 files

# Vaex required us to give data in form of hdf5 format

# In[ ]:


vaex_df = vaex.from_csv(file_path)


# In[ ]:


type(vaex_df)


# ### Read Hdf5 files using Vaex library

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


vaex_df = vaex.open('/kaggle/working/main_dataset.csv')


# In[ ]:


type(vaex_df)


# In[ ]:


vaex_df.head()


# ### Expression system
# - Let's try to implement some expressions using vaex
# - Don't waste memory or time with feature engineering, we (lazily) transform your data when needed.

# In[ ]:


get_ipython().run_cell_magic('time', '', "vaex_df['multiplication_col13']=vaex_df.c1*vaex_df.c3")


# In[ ]:


vaex_df['multiplication_col13']


# ### Out-of-core DataFrame
# Filtering and evaluating expressions will not waste memory by making copies; the data is kept untouched on disk, and will be streamed only when needed. Delay the time before you need a cluster.

# In[ ]:


vaex_df[vaex_df.c2>70]


# #### Filtering will not make a memory copy

# In[ ]:


dff=vaex_df[vaex_df.c2>70]


# #### All the agorithms work out of core, the limit is the size of your hard driver

# In[ ]:


dff.c2.minmax(progress='widget')


# ### Please upvote this introductory notebook on Vaex if you like it! :)
