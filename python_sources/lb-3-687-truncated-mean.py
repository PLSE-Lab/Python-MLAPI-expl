#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


df1 = pd.read_csv("../input/elo-blending/BlendingRLSR.csv")
df1.head()


# In[ ]:


df2 = pd.read_csv("../input/elo-blending/combining_submission (1).csv")
df2.head()

df3 = pd.read_csv("../input/simple-lightgbm-without-blending/submission.csv")
df3.head()

df2['target'] = df2['target'] * 0.35 + df1['target'] * 0.65
df2['target'] = df2['target'] * 0.57 + df3['target'] * 0.43
df2.to_csv("blend.csv",index = False)


# #### Biggest Blending

# In[ ]:


import numpy as np # NUMPY
import pandas as p # PANDAS

# DATA VIZUALIZATION LIBRARIES
from matplotlib import pyplot as plt
import seaborn as sns

# METRICS TO MEASURE RMSE
from math import sqrt
from sklearn import metrics


# In[ ]:


#ALL PUBLIC SOLUTION RMSE < 0.2269 (WITHOUT REPETITIONS)
df_base0 = p.read_csv('../input/elo-blending/3.695.csv',names=["card_id","target0"], skiprows=[0],header=None)
df_base1 = p.read_csv('../input/elo-blending/3.696.csv',names=["card_id","target1"], skiprows=[0],header=None)
df_base2 = p.read_csv('../input/elo-blending/3.6999.csv',names=["card_id","targe2"], skiprows=[0],header=None)
df_base3 = p.read_csv('../input/elo-blending/3.69991.csv',names=["card_id","target3"], skiprows=[0],header=None)
df_base4 = p.read_csv('../input/elo-blending/3.699992.csv',names=["card_id","target4"], skiprows=[0],header=None)
df_base5 = p.read_csv('../input/elo-blending/3.70.csv',names=["card_id","target5"], skiprows=[0],header=None)
df_base6 = p.read_csv('../input/elo-blending/3.701.csv',names=["card_id","target6"], skiprows=[0],header=None)
df_base7 = p.read_csv('../input/elo-blending/3.702.csv',names=["card_id","target7"], skiprows=[0],header=None)
df_base8 = p.read_csv('../input/elo-blending/3.703.csv',names=["card_id","target8"], skiprows=[0],header=None)
df_base9 = p.read_csv('../input/elo-blending/3.704.csv',names=["card_id","target9"], skiprows=[0],header=None)
df_base10 = p.read_csv('../input/elo-blending/Blending.csv',names=["card_id","target10"], skiprows=[0],header=None)
df_base11 = p.read_csv('../input/elo-blending/BlendingRLS.csv',names=["card_id","target11"], skiprows=[0],header=None)
df_base12 = p.read_csv('../input/elo-blending/combining_submission (1).csv',names=["card_id","target12"], skiprows=[0],header=None)
df_base13 = p.read_csv('../input/elo-blending/BlendingRLSR.csv',names=["card_id","target13"], skiprows=[0],header=None)
df_base14 = p.read_csv('../input/simple-lightgbm-without-blending/submission.csv',names=["card_id","target14"], skiprows=[0],header=None)


# In[ ]:


df_base = p.merge(df_base12,df_base0,how='inner',on='card_id')
df_base = p.merge(df_base,df_base1,how='inner',on='card_id')
df_base = p.merge(df_base,df_base2,how='inner',on='card_id')
df_base = p.merge(df_base,df_base3,how='inner',on='card_id')
df_base = p.merge(df_base,df_base4,how='inner',on='card_id')
df_base = p.merge(df_base,df_base5,how='inner',on='card_id')
df_base = p.merge(df_base,df_base6,how='inner',on='card_id')
df_base = p.merge(df_base,df_base7,how='inner',on='card_id')
df_base = p.merge(df_base,df_base8,how='inner',on='card_id')
df_base = p.merge(df_base,df_base9,how='inner',on='card_id')
df_base = p.merge(df_base,df_base10,how='inner',on='card_id')
df_base = p.merge(df_base,df_base11,how='inner',on='card_id')
df_base = p.merge(df_base,df_base12,how='inner',on='card_id')
df_base = p.merge(df_base,df_base13,how='inner',on='card_id')
df_base = p.merge(df_base,df_base14,how='inner',on='card_id')


# In[ ]:


#CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
plt.figure(figsize=(16,12))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")


# In[ ]:


# ALTERNATIVE WAY - RMSE MATRIX (RMSE to measure how similar are 2 solutions)
M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
    for j in np.arange(M.shape[1]):
        M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))


# In[ ]:


#SOLUTION = MEAN OF COLUMNS
df_base['target'] = df_base.iloc[:,1:].mean(axis=1)
df_base[['card_id','target']].to_csv("Bestoutput.csv",index=False)


# # We have take less correlation columns to blend data

# In[ ]:


df_base14 = p.read_csv('../input/simple-lightgbm-without-blending/submission.csv',names=["card_id","target14"], skiprows=[0],header=None)
# df_base5 = p.read_csv('../input/elo-blending/3.70.csv',names=["card_id","target5"], skiprows=[0],header=None)
df_base6 = p.read_csv('../input/elo-blending/3.701.csv',names=["card_id","target6"], skiprows=[0],header=None)
df_base7 = p.read_csv('../input/elo-blending/3.702.csv',names=["card_id","target7"], skiprows=[0],header=None)
# df_base8 = p.read_csv('../input/elo-blending/3.703.csv',names=["card_id","target8"], skiprows=[0],header=None)

df_base = p.merge(df_base12,df_base6,how='inner',on='card_id')
# df_base = p.merge(df_base,df_base5,how='inner',on='card_id')
df_base = p.merge(df_base,df_base7,how='inner',on='card_id')
# df_base = p.merge(df_base,df_base7,how='inner',on='card_id')
# df_base = p.merge(df_base,df_base8,how='inner',on='card_id')
#CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
plt.figure(figsize=(16,12))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")


# In[ ]:


# ALTERNATIVE WAY - RMSE MATRIX (RMSE to measure how similar are 2 solutions)
M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
    for j in np.arange(M.shape[1]):
        M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))


# In[ ]:


#SOLUTION = MEAN OF COLUMNS
df_base['target'] = df_base.iloc[:,1:].mean(axis=1)
df_base[['card_id','target']].to_csv("blend2.csv",index=False)


# In[ ]:


df_base['target'] = df2['target']* 0.3 + df_base['target'] * 0.7

plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
sns.boxplot(df2['target'],orient='v')

plt.subplot(1, 2, 2)
sns.boxplot(df_base['target'], orient='v')
plt.show()
# df_base[['card_id','target']].to_csv("blend3.csv",index=False)


# In[ ]:


from scipy.stats import truncnorm


# In[ ]:


df_base['target'] = truncnorm.mean(df2['target'],df_base['target'])
df_base[['card_id','target']].to_csv("blend3.csv",index=False)


# In[ ]:


display(df_base['target'].head())
plt.figure(figsize=(15,8))
sns.boxplot(df_base['target'], orient='h')
plt.show()

