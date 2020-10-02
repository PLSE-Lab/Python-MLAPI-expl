#!/usr/bin/env python
# coding: utf-8

# ## Little Faster IO : Use pickle format
# 
# >** Use only before last submission. Because this is Kernel Only Competion.**
# 
# **Next Content**
# 
# - [Little Faster IO : Use pickle format(2)](https://www.kaggle.com/subinium/little-faster-io-use-pickle-format-2)
# 
# The train data of this contest is too big(3GB).
# I share the trick to speed up I/O a bit.

# ## What is Pickle?
# 
# - [pickle - Python object serialization](https://docs.python.org/3/library/pickle.html)
# 
# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
# 
# 

# In[ ]:


import pandas as pd
import pickle


# The existing input is taken as follows:

# In[ ]:


get_ipython().run_cell_magic('time', '', "PATH = '../input/data-science-bowl-2019'\ntrain_df = pd.read_csv(PATH+'/train.csv')")


# ## Pandas with pickle

# We will use pickle to use a faster format, more suitable for Python I/O.
# 
# use `pd.to_pickle`. (~30sec)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df.to_pickle("./train.pkl")')


# The file can be read as follows:

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_pickle = pd.read_pickle('./train.pkl')")


# You can also see that import is faster here.
# 
# I will show how this can be used in the next kernel.
# 
# - [Little Faster IO : Use pickle format(2)](https://www.kaggle.com/subinium/little-faster-io-use-pickle-format-2)
# 

# In[ ]:




