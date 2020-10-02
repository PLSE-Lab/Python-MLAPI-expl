#!/usr/bin/env python
# coding: utf-8

# @RokoMijicDev was trying to sample from a large Pandas dataframe, but found it was unexpectedly very slow (https://twitter.com/RokoMijicDev/status/1280089196926054400). I decided to take a look for myself.
# 
# Let's start by importing the necessary libraries.

# In[ ]:


import numpy as np
import pandas as pd


# Now let's create some random data to sample from.

# In[ ]:


data = np.random.randint(0, 1000, size=(1000000, 8))


# In[ ]:


df = pd.DataFrame(data=data, columns=list('abcdefgh'))


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.sample(100, replace=False)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.sample(100, replace=True)')


# As expected, sampling with replacement is faster than sampling without replacement, but the difference seems surprisingly high. Let's try the simplest implementation of sampling-without-replacement that's not completely stupid.

# In[ ]:


import random


# In[ ]:


def dumb_sample(df, n):
    chosen = set()
    max_i = df.shape[0] - 1
    size = 0
    while size < n:
        i = random.randint(0, max_i)
        if i not in chosen:
            chosen.add(i)
            size += 1
    return df.iloc[list(chosen)]


# In[ ]:


df2 = dumb_sample(df, 5)
df2


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'dumb_sample(df, 100)')


# Much faster than the native Pandas method - nice! Let's try taking a bigger sample :-)

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.sample(10000, replace=False)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.sample(10000, replace=True)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'dumb_sample(df, 10000)')


# Our dumb sampling function is now comparable to the native one. Let's try an even bigger sample.

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.sample(100000, replace=False)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df.sample(100000, replace=True)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'dumb_sample(df, 100000)')


# Our dumb sampling function is now slower than the native Pandas one. Note how the native function takes roughly the same time no matter the size of our sample, suggesting that it's doing some step which is O(len(df)). Let's try a bigger dataframe!

# In[ ]:


df2 = pd.DataFrame(data=np.random.randint(0, 1000, size=(50000000, 8)), columns=list('abcdefgh'))


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df2.sample(100, replace=False)')


# Yikes.

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df2.sample(100, replace=True)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'dumb_sample(df2, 100)')


# And with a larger sample size...

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df2.sample(100000, replace=False)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df2.sample(100000, replace=True)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'dumb_sample(df2, 100000)')


# All three functions display the scaling behaviour we've come to expect.
# 
# I think the takeaway is clear: using Pandas' native `sample` method to sample without replacement from large dataframes performs horribly, especially when the sample is small.
