#!/usr/bin/env python
# coding: utf-8

# # STEP 1: SETUP

# In[ ]:


import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # SETUP 2: INITIALIZE DATA

# In[ ]:


df = pd.read_csv("/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_year.csv").drop("Unnamed: 0", 1)


# In[ ]:


changes = df.corr()["year"].drop(["key", "mode", "year", "tempo", "loudness", "duration_ms", "popularity"])
changes


# # STEP 3: VISUALIZE ANY WAY AS YOU WANT

# In[ ]:


df[changes.index.tolist()].plot()


# # STEP 4: WORK ON FORECASTING

# In[ ]:




