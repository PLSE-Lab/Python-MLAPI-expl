#!/usr/bin/env python
# coding: utf-8

# Credits to the Experts (Please like their kernels)<br>
# Ashish Gupta: [20+ top lgbm models outputs](https://www.kaggle.com/roydatascience/lgmodels)<br>
# Navaneetha: [xtreme-boost-and-feature-engineering](https://www.kaggle.com/krishonaveen/xtreme-boost-and-feature-engineering)<br>
# Shugen: [lgb-starter-r](https://www.kaggle.com/andrew60909/lgb-starter-r) <br>
# Khan HBK: [hust-lgb-starter-with-r](https://www.kaggle.com/duykhanh99/hust-lgb-starter-with-r)<br>
# Konstantin: [ieee-internal-blend](https://www.kaggle.com/kyakovlev/ieee-internal-blend)<br>
# Avocado: [xgb-model-with-feature-engineering](https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering)<br>
# David: [feature-engineering-lightgbm-w-gpu](https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu)<br>
# Lyalikov: [lgbm-baseline-small-fe-no-blend](https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend)<br>
# Yuanrong: [lgb-xgb-ensemble-stacking-based-on-fea-eng](https://www.kaggle.com/yw6916/lgb-xgb-ensemble-stacking-based-on-fea-eng)<br>
# 

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

from scipy.stats import describe
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Stacking Approach using GMEAN

# In[ ]:


LABELS = ["isFraud"]
all_files = glob.glob("../input/lgmodels/*.csv")
scores = np.zeros(len(all_files))
for i in range(len(all_files)):
    scores[i] = float('.'+all_files[i].split(".")[3])
    print(i,scores[i],all_files[i])


# In[ ]:


describe(scores)


# In[ ]:


top = scores.argsort()[::-1]
for i, f in enumerate(top):
    print(i,scores[f],all_files[f])


# In[ ]:


outs = [pd.read_csv(all_files[f], index_col=0) for f in top]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols


# In[ ]:


# check correlation
corr = concat_sub.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))

# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr,mask=mask,cmap='prism',center=0, linewidths=1,
                annot=True,fmt='.4f', cbar_kws={"shrink":.2})


# # Weighted GMEAN by inverse correlation

# In[ ]:


rank = np.tril(corr.values,-1)
rank[rank<0.92] = 1
m = (rank>0).sum() - (rank>0.97).sum()
m_gmean, s = 0, 0
for n in range(m):
    mx = np.unravel_index(rank.argmin(), rank.shape)
    w = (m-n)/m
    m_gmean += w*(np.log(concat_sub.iloc[:,mx[0]])+np.log(concat_sub.iloc[:,mx[1]]))/2
    s += w
    rank[mx] = 1
m_gmean = np.exp(m_gmean/s)


# In[ ]:


m_gmean = (m_gmean-m_gmean.min())/(m_gmean.max()-m_gmean.min())
describe(m_gmean)


# In[ ]:


concat_sub['isFraud'] = m_gmean
concat_sub[['isFraud']].to_csv('stack_gmean.csv')

