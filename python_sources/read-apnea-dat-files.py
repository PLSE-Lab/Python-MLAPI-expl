#!/usr/bin/env python
# coding: utf-8

# # Reading the Apnea Dat Files
# 
# The data for the apnea dataset is from the [physionet](https://www.physionet.org/physiobank/database/) library and to read it in Python you need the [wfdb](https://www.physionet.org/physiotools/wfdb.shtml) package. This is not available in a the kaggle kernels so this notebook won't work online for now. Feel free to try locally and feed back.

# In[2]:


# !pip install wfdb
import wfdb
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


recordname = "../input/apnea-ecg/a04"

record = wfdb.rdsamp(recordname)


# The main data is the `p_signals` element. The way we've read this in it is the raw analogue data.

# ## Annotations
# The annotation data is where each 60 second chunk has been labelled normal (N) or atrial premature contraction (A)

# In[9]:


annotation = wfdb.rdann(recordname, extension="apn")

annotation.contained_labels


# We can findout which field contains the labels and then get them as a numpy array

# In[10]:


annotation.get_label_fields()


# In[11]:


annotation.symbol[:10]


# In[12]:


np.unique(annotation.symbol, return_counts=True)


# Make a plot

# In[18]:


record_small = wfdb.rdsamp(recordname, sampto = 5999)
annotation_small = wfdb.rdann(recordname, extension="apn", sampto = 5999)
wfdb.plot_all_records("../input/apnea-ecg")

