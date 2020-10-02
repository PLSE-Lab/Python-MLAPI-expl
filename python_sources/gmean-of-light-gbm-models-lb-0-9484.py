#!/usr/bin/env python
# coding: utf-8

# <pre>Credits to the Experts (Please like their kernels)
# Ashish Gupta: https://www.kaggle.com/roydatascience/aggregating-the-light-gbm-models-0-9469
# Navaneetha: https://www.kaggle.com/krishonaveen/xtreme-boost-and-feature-engineering
# Shugen: https://www.kaggle.com/andrew60909/lgb-starter-r
# Khan HBK: https://www.kaggle.com/duykhanh99/hust-lgb-starter-with-r 
# Konstantin: https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again/output
# Avocado: https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering
# David: https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
# Lyalikov: https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
# Yuanrong: https://www.kaggle.com/yw6916/lgb-xgb-ensemble-stacking-based-on-fea-eng
# </pre>

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
all_files


# In[ ]:


outs = [pd.read_csv(f, index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)


# In[ ]:


# check correlation
corr = concat_sub.iloc[:,1:].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,mask=mask,cmap='prism',vmin=0.95,center=0,linewidths=1,annot=True,fmt='.4f')


# # Weighted GMEAN by inverse correlation

# In[ ]:


rank = np.tril(concat_sub.iloc[:,1:].corr().values,-1)
m = (rank>0).sum()
m_gmean, s = 0, 0
for n in range(min(rank.shape[0],m)):
    mx = np.unravel_index(rank.argmin(), rank.shape)
    w = (m-n)/(m+n)
    print(w)
    m_gmean += w*(np.log(concat_sub.iloc[:,mx[0]+1])+np.log(concat_sub.iloc[:,mx[1]+1]))/2
    s += w
    rank[mx] = 1
m_gmean = np.exp(m_gmean/s)


# In[ ]:


describe(m_gmean)


# In[ ]:


concat_sub['isFraud'] = m_gmean
concat_sub[['TransactionID','isFraud']].to_csv('stack_gmean.csv', 
                                        index=False, float_format='%.4g')

