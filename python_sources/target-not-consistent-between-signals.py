#!/usr/bin/env python
# coding: utf-8

# Hi there,
# 
# Is it ok that the same measurement have different target labels between signals?
# According to data description it should be the same (or not really?) . There are 38 cases of measurements with not consistent labels between signals.
# 
# Quick and dirty code to show the problem below:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/metadata_train.csv')
df.info()


# In[ ]:


df.head()


# In[ ]:


#let's check if targets are consistent within the same measurement id
targets = df.groupby('id_measurement')[['target','id_measurement']].agg('mean')
targets.head()


# In[ ]:


sns.countplot(x='target',data=targets)
# it should be only "1" and "0" but we have cases where target is not consitent 


# In[ ]:


mislabeled = targets.loc[(targets.target <1 ) & (targets.target > 0.3) ,'id_measurement']
print(str(mislabeled.shape[0]) + ' measurments most likely mislabeled' )


# In[ ]:


# qc it all


# In[ ]:



df.loc[df.id_measurement.isin(mislabeled) ,:]


# In[ ]:




