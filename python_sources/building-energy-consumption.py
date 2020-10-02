#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/buildingdata.csv')
#test = pd.read_csv('../input/test.csv')
train.head(2)


# In[ ]:


train.describe()


# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(20,9))
sns.heatmap(corrmat, vmax=0.8, annot=True)


# In[ ]:


corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat['Total electricity consumption'])>0.5]
plt.figure(figsize=(10,10))
sns.heatmap(train[top_corr_features].corr(), annot=True, cmap = 'RdYlGn')


# In[ ]:


corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat['Total electricity consumption'])>0.8]
plt.figure(figsize=(7,7))
sns.heatmap(train[top_corr_features].corr(), annot=True, cmap = 'RdYlGn')


# In[ ]:


corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat['Total electricity consumption'])>0.9]
plt.figure(figsize=(5,5))
sns.heatmap(train[top_corr_features].corr(), annot=True, cmap = 'RdYlGn')


# In[ ]:




