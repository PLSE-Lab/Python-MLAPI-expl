#!/usr/bin/env python
# coding: utf-8

# **Load tools**

# In[ ]:


#data analysis
import pandas as pd
import numpy as np
#visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Machine learing
from sklearn.ensemble import RandomForestClassifier


# **Import data**

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv') 


# **Preview**

# In[ ]:


#values
print(train_df.columns.values)

#head
train_df.head()


# 
