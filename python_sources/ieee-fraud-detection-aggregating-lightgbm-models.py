#!/usr/bin/env python
# coding: utf-8

# While I navigated the Blending Kernels available for this competition. I always fear about overfitting on Private Leaderboard.  While I am still finalizing my kernel with EDA and Model, The inputs that use in this kernel are generated from the great contributions made by other Kernel GrandMasters, Masters or Experts. None of my inputs contains blends or stack results.

# <pre><b>Credits to the Experts (Please like their kernels)
# 1. Navaneetha Kernel : https://www.kaggle.com/krishonaveen/xtreme-boost-and-feature-engineering
# 2. Shugen Kernel : https://www.kaggle.com/andrew60909/lgb-starter-r
# 3. Khan HBK Kernel : https://www.kaggle.com/duykhanh99/hust-lgb-starter-with-r 
# 4. Konstantin Kernel : https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again/output
# 5. Avocado Kernel : https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering?scriptVersionId=18686303
# 6. David's Kernel : https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu
# 7. Lyalikov's Kernel : https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
# 8. Yuanrong's Kernel : https://www.kaggle.com/yw6916/lgb-xgb-ensemble-stacking-based-on-fea-eng
# 9. Steve's Kernel : https://www.kaggle.com/abednadir/best-r-score
# </b></pre>

# In[ ]:


from scipy.stats import rankdata
from scipy.stats.mstats import gmean

LABELS = ["isFraud"]


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import glob
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
all_files = glob.glob("../input/lgmodels/*.csv")
all_files

# Any results you write to the current directory are saved as output.


# In[ ]:


predict_list = []
predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9433.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9451.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9459.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9463.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9467.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9440.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9454.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9466.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9475.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9433.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9468.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9452.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9429.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9449.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9457.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9438.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9442.csv')[LABELS].values)
predict_list.append(pd.read_csv('../input/lgmodels/submission-.9469.csv')[LABELS].values)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(1):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
submission[LABELS] = predictions
submission.to_csv('AggStacker.csv', index=False)


# In[ ]:


submission.head()


# # Stacking Approach using GMean and Median

# <pre><b>The idea of GMean is taken from Paulo's Kernel https://www.kaggle.com/paulorzp/gmean-of-light-gbm-models-lb-0-947x</b></pre>

# In[ ]:


sub_path = "../input/lgmodels/"
all_files = os.listdir(sub_path)
all_files


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


# check correlation
concat_sub.iloc[:,1:ncol].corr()


# In[ ]:


concat_sub.describe()


# In[ ]:


corr = concat_sub.iloc[:,1:7].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


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


# # Geometric Mean Stacking

# In[ ]:


# get the data fields ready for stacking
concat_sub['isFraud'] = m_gmean
concat_sub[['TransactionID','isFraud']].to_csv('stack_gmean.csv', 
                                        index=False, float_format='%.4g')


# # Median Stacking

# In[ ]:


concat_sub['m_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub['isFraud'] = concat_sub['m_median']
concat_sub[['TransactionID','isFraud']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')


# AggStacker.csv generates the best score (.9475)
