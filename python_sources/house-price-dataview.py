#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.info()


# In[ ]:


plt.rcParams['font.size']=12


# In[ ]:


sample_data=df_train['SalePrice']
sample_label=df_train['Id']


# In[ ]:


sample_df = pd.DataFrame(columns=["label", "SalePrice"])
sample_df


# In[ ]:


sample_df["label"] = df_train['Id']
sample_df["SalePrice"]=df_train['SalePrice']


# In[ ]:


sample_df


# In[ ]:


sample_df = sample_df.sort_values(by="SalePrice", ascending=False)
sample_df


# In[ ]:


sample_df["SalePrice_sum"] = np.cumsum(sample_df["SalePrice"])
sample_df


# In[ ]:


sample_df["accum_percent"] = sample_df["SalePrice_sum"] / sum(sample_df["SalePrice"]) * 100
sample_df


# In[ ]:


fig, ax1 = plt.subplots(figsize=(6,4))
data_num = len(sample_df)

ax1.bar(range(data_num), sample_df["SalePrice"])
ax1.set_xticks(range(data_num))
ax1.set_xticklabels(sample_df["label"].tolist())
ax1.set_xlabel("label")
ax1.set_ylabel("price")

ax2 = ax1.twinx()
ax2.plot(range(data_num), sample_df["accum_percent"], c="k", marker="o")
ax2.set_ylim([0, 100])

ax2.grid(True, which='both', axis='y')

ax1.set_title("SalePrice")


# In[ ]:


plt.show()


# In[ ]:


df_train.columns


# In[ ]:


sns.distplot(df_train['SalePrice']);


# In[ ]:


var='OverallQual'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
f,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=var,y="SalePrice",data=data)
fig.axis(ymin=0,ymax=800000);


# In[ ]:


sns.pairplot(df_train,hue="SalePrice")


# In[ ]:


sns.jointplot(df_train.columns[19],df_train.columns[80],df_train,kind="reg")


# In[ ]:


sns.jointplot(df_train.columns[19],df_train.columns[80],df_train,kind="hex")


# In[ ]:


sns.heatmap(df_train.corr())


# In[ ]:




