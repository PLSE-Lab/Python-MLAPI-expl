#!/usr/bin/env python
# coding: utf-8

# # Renaming features to super heroes "docker style"
# 
# I am probably not the only one who is appaled by all the tons of those auto-generated feature names. So I made a simple kernel
# 
# As a bonus, I am doing some exploratory plotting on those features, which tells me something everybody already knows:
#     - there are a not of features
#     - they are full of zeros

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")
test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")


# In[ ]:


#generating names
def get_names(n_names):
    epithets = ["adoring","amazing","angry",
                "blissful","boring","brave",
                "clever","cocky","compassionate"]
    heroes = pd.read_csv("../input/superhero-set/heroes_information.csv")
    heronames = heroes.name.str.lower().str.replace("\s+", "_").str.replace("-+","_").unique()
    namelist = [epi + "_" + nm for epi in epithets for nm in heronames]
    return namelist[:n_names]
new_cols =  get_names(len(train.columns) - 2)
col_map = {c:v for c, v in zip(test.columns.tolist()[1:], new_cols  )}
train.columns = ["ID", "target"] + new_cols
test.columns = ["ID"] + new_cols


# In[ ]:


colmap_df = pd.DataFrame(pd.Series(col_map))
colmap_df.to_csv("col_map.csv",index=True)


# In[ ]:


colmap_df.head()


# # Looking at features` descriptive stats and plotting distributions of those stats for all the features

# In[ ]:


train_desc = train.describe().T


# In[ ]:


train_desc.head()


# In[ ]:


test_desc = test.describe().T


# In[ ]:


test_desc.head()


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


def plot_dist(varname):
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.title(f"Distribution of variable {varname} in train")
    sns.distplot(train_desc[varname])
    plt.title(f"Distribution of {varname}  in test")
    plt.subplot(2,1,2)
    sns.distplot(test_desc[varname])
for varname in train_desc.columns[1:]:
    plot_dist(varname)


# In[ ]:


sns.distplot(train.target)


# In[ ]:


train.dtypes


# In[ ]:


train["adoring_adam_strange"][train["adoring_adam_strange"] > 0]


# In[ ]:


train["clever_venom"][train["clever_venom"] > 0]


# In[ ]:


sns.distplot(np.log1p(train.target))

