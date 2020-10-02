#!/usr/bin/env python
# coding: utf-8

# # An easy and generic method to blend kernels
# 
# ## planning to provide a more thorough approach soon!

# ## imports

# In[ ]:


import os
import sys
import time
import numpy as np
import pandas as pd 
import seaborn as sns
from math import sqrt
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# ## getting the data

# In[ ]:


df1 = pd.read_csv("../input/elo-blending/3.695.csv")
df2 = pd.read_csv("../input/elo-blending/3.696.csv")
df3 = pd.read_csv("../input/submit/submit.csv")
df4 = pd.read_csv("../input/combined/combining_submission.csv")


# ## reading several high scores

# In[ ]:


df_base0  = pd.read_csv('../input/elo-blending/3.695.csv',      names=["card_id","target0"],  skiprows=[0],header=None)
df_base1  = pd.read_csv('../input/elo-blending/3.696.csv',      names=["card_id","target1"],  skiprows=[0],header=None)
df_base2  = pd.read_csv('../input/elo-blending/3.6999.csv',     names=["card_id","target2"],  skiprows=[0],header=None)
df_base3  = pd.read_csv('../input/elo-blending/3.69991.csv',    names=["card_id","target3"],  skiprows=[0],header=None)
df_base4  = pd.read_csv('../input/elo-blending/3.699992.csv',   names=["card_id","target4"],  skiprows=[0],header=None)
df_base5  = pd.read_csv('../input/elo-blending/3.70.csv',       names=["card_id","target5"],  skiprows=[0],header=None)
df_base6  = pd.read_csv('../input/elo-blending/3.701.csv',      names=["card_id","target6"],  skiprows=[0],header=None)
df_base7  = pd.read_csv('../input/elo-blending/3.702.csv',      names=["card_id","target7"],  skiprows=[0],header=None)
df_base8  = pd.read_csv('../input/elo-blending/3.703.csv',      names=["card_id","target8"],  skiprows=[0],header=None)
df_base9  = pd.read_csv('../input/elo-blending/3.704.csv',      names=["card_id","target9"],  skiprows=[0],header=None)
df_base10 = pd.read_csv('../input/elo-blending/Blending.csv',   names=["card_id","target10"], skiprows=[0],header=None)
df_base11 = pd.read_csv('../input/elo-blending/BlendingRLS.csv',names=["card_id","target11"], skiprows=[0],header=None)


# ## merging all the solutions into one

# In[ ]:


df_base = pd.merge(df_base0,df_base1,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base2,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base3,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base4,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base5,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base6,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base7,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base8,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base9,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base10,how='inner',on='card_id')
df_base = pd.merge(df_base,df_base11,how='inner',on='card_id')


# ## creating a heatmap to understand easier

# In[ ]:


plt.figure(figsize=(16,12))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")


# ## our metric is mse

# In[ ]:


M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
    for j in np.arange(M.shape[1]):
        M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))


# ## let's find the median

# In[ ]:


df_base_median = df_base.iloc[:,1:].median(axis=1)


# In[ ]:


df_base0  = pd.read_csv('../input/elo-blending/3.695.csv',      names=["card_id","target0"],  skiprows=[0],header=None)
df_base1  = pd.read_csv('../input/elo-blending/3.696.csv',      names=["card_id","target1"],  skiprows=[0],header=None)
df_base10 = pd.read_csv('../input/elo-blending/Blending.csv',   names=["card_id","target10"], skiprows=[0],header=None)
df_base11 = pd.read_csv('../input/elo-blending/BlendingRLS.csv',names=["card_id","target11"], skiprows=[0],header=None)

df_base = pd.merge(df_base0,df_base1,how='inner',on='card_id')
df_base = pd.merge(df_base0,df_base10,how='inner',on='card_id')
df_base = pd.merge(df_base0,df_base11,how='inner',on='card_id')

plt.figure(figsize=(12,8))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")


# In[ ]:


M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
    for j in np.arange(M.shape[1]):
        M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))


# In[ ]:


df_base['target'] = df_base_median
df_base['target4'] = df4['target']


# ## a tiny post-process trick
# ## checking the outliers in train and comparing to test/

# In[ ]:


df_final = np.zeros(len(df_base))
a=-10*np.log2(10)
thresh = -14

for i in range(len(df3)-1):
    if df3['target'][i]< thresh:
        df_final[i]=a
    else:
        df_final[i]=df_base['target'][i]
pd.Series(df_final).value_counts().head(1)


# ## kinda normalizing

# In[ ]:


magic = np.median(df_final) - df_final.std()


# In[ ]:


for i in range(len(df_final)-1):
    if df_final[i] > magic:
        df_final[i] =  df_final[i] - abs(np.median(df_final))/8 +0.001


# ## and submission

# In[ ]:


df_finall=pd.DataFrame(df_base['card_id'])
df_finall['target'] = df_final
df_finall[['card_id','target']].to_csv("good_output.csv",index=False)


# ## thank you very much, please upvote if you find it useful!
