#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## 1. Take a look at effect of joint distribution on Targets

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')


# In[ ]:


def corr_plot(v1, v2, n=10):
    new = pd.DataFrame()
    new[v1] = pd.cut(train_df[v1], 10, labels=[v1 + '_' + str(i) for i in range(1, 11)])
    new[v2] = pd.cut(train_df[v2], 10, labels=[v2 + '_' + str(i) for i in range(1, 11)])
    new['target'] = train_df['target']
    new = new.groupby([v1, v2])['target'].mean().reset_index()
    new = pd.pivot_table(index=[v1], columns=[v2], values=['target'], data=new)
#     plt.figure(figsize=(10,8))
    sns.heatmap(new)
    plt.show()
    return new


# In[ ]:


corr_plot('var_108', 'var_154')


# ## It seems work, some joint encode have a high targets mean value.
# ## 2. Let's check the information gains from joint encode.

# In[ ]:


df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
# add variable category decode
df_decode = pd.DataFrame()
n_cut = 10
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
for col in features:
    col_decode = col + '_' + 'decode'
    df_decode[col_decode] = pd.cut(df[col], n_cut, labels=range(0, n_cut))


# In[ ]:


def add_joint(v1, v2):
    return df_decode[v1 + '_' + 'decode'].astype(str) + df_decode[v2 + '_' + 'decode'].astype(str)


# In[ ]:


def entropy(x):
    uniq, counts = np.unique(x, return_counts=True)
    uniq_prob = counts / counts.sum()
    entr = -np.sum(uniq_prob * np.log2(uniq_prob))
    return entr

def condEntropy(cond, target):
    cond_df = pd.DataFrame({'cond': cond, 'target': target}).dropna()
    entr = cond_df.groupby('cond')['target'].apply(entropy)
    prob = cond_df.groupby('cond')['target'].apply(lambda x: x.count() / cond_df.shape[0])
    return np.sum(entr * prob)


# In[ ]:


origin_target_entropy = entropy(train_df.target)
var108_cond_entropy = condEntropy(df_decode['var_108_decode'].iloc[:200000], train_df.target)
var154_cond_entropy = condEntropy(df_decode['var_154_decode'].iloc[:200000], train_df.target)
joint_cond_entropy = condEntropy(add_joint('var_108', 'var_154').iloc[:200000], train_df.target)


# In[ ]:


print(origin_target_entropy, var108_cond_entropy, var154_cond_entropy, joint_cond_entropy)


# In[ ]:





# ## emmmm  have a higher information gain indeed
# ## 3. find the Top K combination

# In[ ]:


from itertools import combinations
combs = list(combinations([col for col in train_df if col not in ['ID_code', 'target']], 2))


# In[ ]:


print(combs[0])
print(len(combs))


# In[ ]:


get_ipython().run_cell_magic('time', '', "joint_cond_entropy = condEntropy(add_joint('var_108', 'var_154').iloc[:200000], train_df.target)")


# ### This will take nearly 3.5 hours

# In[ ]:


df_decode = df_decode.iloc[:2000000]


# In[ ]:


get_ipython().run_cell_magic('time', '', "from tqdm import tqdm\n\nresult = dict()\nfor c in tqdm(combs):\n    ce  = condEntropy(add_joint(c[0], c[1]), train_df.target)\n    result[c] = ce\nresult = pd.Series(result).reset_index()\nresult.columns = ['v1', 'v2', 'entropy']\nresult = result.sort_values('entropy', ascending=True)\nresult.to_csv('joint_cond_entropy.csv')")


# In[ ]:




