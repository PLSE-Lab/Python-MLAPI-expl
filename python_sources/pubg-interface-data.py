#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[ ]:


# Set the size of the plots 
plt.rcParams["figure.figsize"] = (18,8)
sns.set(rc={'figure.figsize':(18,8)})


# In[ ]:


data = pd.read_csv("../input/pubg-presentation-features-engineering/train.csv")
print("Finished loading the data")


# In[ ]:


data = data[['walkDistance', 'swimDistance', 'rideDistance', 'matchDuration', 'kills', 'killPlace', 'maxPlace', 'numGroups', 'winPlacePerc']]


# In[ ]:


data.head(20)


# In[ ]:




