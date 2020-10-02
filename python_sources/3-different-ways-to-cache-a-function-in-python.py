#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import functools
import joblib
import time

import numpy as np


# In[ ]:


get_ipython().system('mkdir /kaggle/working/joblib_cache/')


# In[ ]:


memory = joblib.Memory("/kaggle/working/joblib_cache/", verbose=0)


# In[ ]:


DICTIONARY_CACHE = {}


# In[ ]:


def long_running_function(number):
    time.sleep(3)
    if number >= 0:
        return np.sqrt(number)
    else:
        return number


# In[ ]:


def dict_cache_predict(number):
    if number in DICTIONARY_CACHE:
        return DICTIONARY_CACHE[number]
    else:
        result = long_running_function(number)
        DICTIONARY_CACHE[number] = result
        return result


# In[ ]:


@functools.lru_cache(maxsize=128)
def long_running_function_functools(number):
    time.sleep(3)
    if number >= 0:
        return(np.sqrt(number))
    else:
        return number


# In[ ]:


@memory.cache
def long_running_function_joblib(number):
    time.sleep(3)
    if number >= 0:
        return(np.sqrt(number))
    else:
        return number


# In[ ]:


list_of_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list_of_numbers = list_of_numbers * 10000


# In[ ]:


# This is the original function. Its going to take atleast 30*10000 seconds
#%time
#for num in list_of_numbers:
#    _ = long_running_function(num)


# In[ ]:


get_ipython().run_line_magic('time', '')
for num in list_of_numbers:
    _ = dict_cache_predict(num)


# In[ ]:


get_ipython().run_line_magic('time', '')
for num in list_of_numbers:
    _ = long_running_function_functools(num)


# In[ ]:


get_ipython().run_line_magic('time', '')
for num in list_of_numbers:
    _ = long_running_function_joblib(num)


# In[ ]:


DICTIONARY_CACHE

