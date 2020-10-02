#!/usr/bin/env python
# coding: utf-8

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
df_base0 = p.read_csv('../input/public-solutions/base0_0.2211.csv',names=["item_id","deal_probability0"], skiprows=[0],header=None)
df_base1 = p.read_csv('../input/public-solutions/base1_0.2212.csv',names=["item_id","deal_probability1"], skiprows=[0],header=None)
df_base2 = p.read_csv('../input/public-solutions/base2_0.2212.csv',names=["item_id","deal_probability2"], skiprows=[0],header=None)
df_base3 = p.read_csv('../input/public-solutions/base3_0.2213.csv',names=["item_id","deal_probability3"], skiprows=[0],header=None)
df_base4 = p.read_csv('../input/public-solutions/base4_0.2215.csv',names=["item_id","deal_probability4"], skiprows=[0],header=None)
df_base5 = p.read_csv('../input/public-solutions/base5_0.2219.csv',names=["item_id","deal_probability5"], skiprows=[0],header=None)
df_base6 = p.read_csv('../input/public-solutions/base6_0.2220.csv',names=["item_id","deal_probability6"], skiprows=[0],header=None)
df_base7 = p.read_csv('../input/public-solutions/base7_0.2222.csv',names=["item_id","deal_probability7"], skiprows=[0],header=None)
df_base8 = p.read_csv('../input/public-solutions/base8_0.2224.csv',names=["item_id","deal_probability8"], skiprows=[0],header=None)
df_base9 = p.read_csv('../input/public-solutions/base9_0.2226.csv',names=["item_id","deal_probability9"], skiprows=[0],header=None)
df_base10 = p.read_csv('../input/public-solutions/base10_0.2227.csv',names=["item_id","deal_probability10"], skiprows=[0],header=None)
df_base11 = p.read_csv('../input/public-solutions/base11_0.2228.csv',names=["item_id","deal_probability11"], skiprows=[0],header=None)
df_base12 = p.read_csv('../input/public-solutions/base12_0.2230.csv',names=["item_id","deal_probability12"], skiprows=[0],header=None)
df_base13 = p.read_csv('../input/public-solutions/base13_0.2232.csv',names=["item_id","deal_probability13"], skiprows=[0],header=None)
df_base14 = p.read_csv('../input/public-solutions/base14_0.2237.csv',names=["item_id","deal_probability14"], skiprows=[0],header=None)
df_base15 = p.read_csv('../input/public-solutions/base15_0.2237.csv',names=["item_id","deal_probability15"], skiprows=[0],header=None)
df_base16 = p.read_csv('../input/public-solutions/base16_0.2238.csv',names=["item_id","deal_probability16"], skiprows=[0],header=None)
df_base17 = p.read_csv('../input/public-solutions/base17_0.2239.csv',names=["item_id","deal_probability17"], skiprows=[0],header=None)
df_base18 = p.read_csv('../input/public-solutions/base18_0.2246.csv',names=["item_id","deal_probability18"], skiprows=[0],header=None)
df_base19 = p.read_csv('../input/public-solutions/base19_0.2247.csv',names=["item_id","deal_probability19"], skiprows=[0],header=None)
df_base20 = p.read_csv('../input/public-solutions/base20_0.2249.csv',names=["item_id","deal_probability20"], skiprows=[0],header=None)
df_base21 = p.read_csv('../input/public-solutions/base21_0.2255.csv',names=["item_id","deal_probability21"], skiprows=[0],header=None)
df_base22 = p.read_csv('../input/public-solutions/base22_0.2255.csv',names=["item_id","deal_probability22"], skiprows=[0],header=None)
df_base23 = p.read_csv('../input/public-solutions/base23_0.2269.csv',names=["item_id","deal_probability23"], skiprows=[0],header=None)


# In[ ]:


#CREATING SOLUTIONS COLUMNS
df_base = p.merge(df_base0,df_base1,how='inner',on='item_id')
df_base = p.merge(df_base,df_base2,how='inner',on='item_id')
df_base = p.merge(df_base,df_base3,how='inner',on='item_id')
df_base = p.merge(df_base,df_base4,how='inner',on='item_id')
df_base = p.merge(df_base,df_base5,how='inner',on='item_id')
df_base = p.merge(df_base,df_base6,how='inner',on='item_id')
df_base = p.merge(df_base,df_base7,how='inner',on='item_id')
df_base = p.merge(df_base,df_base8,how='inner',on='item_id')
df_base = p.merge(df_base,df_base9,how='inner',on='item_id')
df_base = p.merge(df_base,df_base10,how='inner',on='item_id')
df_base = p.merge(df_base,df_base11,how='inner',on='item_id')
df_base = p.merge(df_base,df_base12,how='inner',on='item_id')
df_base = p.merge(df_base,df_base13,how='inner',on='item_id')
df_base = p.merge(df_base,df_base14,how='inner',on='item_id')
df_base = p.merge(df_base,df_base15,how='inner',on='item_id')
df_base = p.merge(df_base,df_base16,how='inner',on='item_id')
df_base = p.merge(df_base,df_base17,how='inner',on='item_id')
df_base = p.merge(df_base,df_base18,how='inner',on='item_id')
df_base = p.merge(df_base,df_base19,how='inner',on='item_id')
df_base = p.merge(df_base,df_base20,how='inner',on='item_id')
df_base = p.merge(df_base,df_base21,how='inner',on='item_id')
df_base = p.merge(df_base,df_base22,how='inner',on='item_id')
df_base = p.merge(df_base,df_base23,how='inner',on='item_id')


# In[ ]:


#CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
plt.figure(figsize=(20,20))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")


# In[ ]:


#ALTERNATIVE WAY - RMSE MATRIX (RMSE to measure how similar are 2 solutions)
M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
 for j in np.arange(M.shape[1]):
    M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(M,annot=True,fmt=".3f")


# #Solutions selection:
# We would like to choose solutions with low correlation and high scores (It is a trade off Correlation vs Score)

# **deal_probability0 (0.2211) - Best Solution
# 
# **deal_probability1 (0.2212) - Second Solution
# 
# **deal_probability2 (0.2212) - Third Solution
# 
# deal_probability3 (0.2213) - Similar to deal_probability0
# 
# deal_probability4 (0.2215) - Similar to deal_probability0
# 
# deal_probability5 (0.2219) - Similar to deal_probability0
# 
# deal_probability6 (0.2220) - Similar to deal_probability0
# 
# deal_probability7 (0.2222) - Similar to deal_probability0
# 
# deal_probability8 (0.2224) - Similar to deal_probability0
# 
# deal_probability9 (0.2226) - Similar to deal_probability0
# 
# deal_probability10 (0.2227) - Similar to deal_probability0
# 
# deal_probability11 (0.2228) - Similar to deal_probability0
# 
# deal_probability12 (0.2230) - Similar to deal_probability0
# 
# deal_probability13 (0.2232) - Similar to deal_probability0
# 
# **deal_probability14 (0.2237) - Low Correlation with deal_probability0**
# 
# deal_probability15 (0.2237) - Similar to deal_probability14
# 
# deal_probability16 (0.2238) -Similar to deal_probability0
# 
# deal_probability17 (0.2239) - Similar to deal_probability0
# 
# **deal_probability18 (0.2246)- Low Correlation with deal_probability0**
# 
# deal_probability19 (0.2247) - Similar to deal_probability0 and not good Score (0.2247)
# 
# deal_probability20 (0.2249) - Similar to deal_probability0 and not good Score (0.2249)
# 
# deal_probability21 (0.2255) - Low Correlation but Not good Score (0.2255)
# 
# deal_probability22 (0.2255) - Low Correlation Not good Score (0.2255)
# 
# deal_probability23 (0.2269) - Low Correlation Not good Score (0.2269)

# In[ ]:


#PORTFOLIO # 0.2204 (0,1,2,14 and 18)
df_base = p.merge(df_base0,df_base1,how='inner',on='item_id')
df_base = p.merge(df_base,df_base2,how='inner',on='item_id')
df_base = p.merge(df_base,df_base14,how='inner',on='item_id')
df_base = p.merge(df_base,df_base18,how='inner',on='item_id')


# In[ ]:


#CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
plt.figure(figsize=(10,10))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")


# In[ ]:


#ALTERNATIVE WAY - RMSE MATRIX (RMSE to measure how similar are 2 solutions)
M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
 for j in np.arange(M.shape[1]):
    M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(M,annot=True,fmt=".3f")


# In[ ]:


#SOLUTION = MEAN OF COLUMNS
df_base['deal_probability'] = df_base.iloc[:,1:].mean(axis=1)


# In[ ]:


#GENERATING FINAL SOLUTION
df_base[['item_id','deal_probability']].to_csv("best_public_blend.csv",index=False)

