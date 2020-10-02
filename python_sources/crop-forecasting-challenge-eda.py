#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#df_train = pd.read_table('../input/TrainingDataSet_Wheat2.csv')
#df_train = pd.read_table('../input/TestDataSet_Wheat_blind.csv')

df_train = pd.read_table('../input/TrainingDataSet_Maize.csv', index_col=0)
df_test_to_submit = pd.read_table('../input/TestDataSet_Maize_blind.csv')


# In[ ]:


df_train.describe()


# In[ ]:


df_test_to_submit.describe()


# In[ ]:


sns.distplot(df_train['yield_anomaly']);


# In[ ]:


sns.distplot(df_train['NUMD']);


# In[ ]:


sns.distplot(df_train['year_harvest']);


# In[ ]:


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


#yield_anomaly correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'yield_anomaly')['yield_anomaly'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(8, 6))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


sns.set()
cols = cols = ['yield_anomaly', 'PR_7', 'SeqPR_7', 'SeqPR_8', 'PR_8', 'PR_6', 'Tn_2']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# In[ ]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:




