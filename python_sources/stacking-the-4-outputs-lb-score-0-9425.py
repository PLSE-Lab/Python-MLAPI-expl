#!/usr/bin/env python
# coding: utf-8

# >* Based on https://www.kaggle.com/lpachuong/statstack
# >* Thanks to <br>
# https://www.kaggle.com/jazivxt/safe-box<br>
# https://www.kaggle.com/artgor/eda-and-models<br>
# https://www.kaggle.com/stocks/under-sample-with-multiple-runs<br>
# https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb

# ## Upvote if this was helpful

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


sub_path = "../input/ieeesubmissions"
all_files = os.listdir(sub_path)
all_files


# In[ ]:


all_files.remove('blend02.csv')
all_files


# In[ ]:


outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "ieee" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


# check correlation
concat_sub.iloc[:,1:ncol].corr()


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


# get the data fields ready for stacking
concat_sub['ieee_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub['ieee_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub['ieee_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub['ieee_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)


# In[ ]:


concat_sub.describe()


# In[ ]:


cutoff_lo = 0.8
cutoff_hi = 0.2


# # Mean Stacking

# In[ ]:


concat_sub['isFraud'] = concat_sub['ieee_mean']
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_mean.csv', 
                                        index=False, float_format='%.6f')


# # Median Stacking

# In[ ]:


concat_sub['isFraud'] = concat_sub['ieee_median']
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')


# # Pushout + Median Stacking
# >* Pushout strategy is bit aggresive

# In[ ]:


concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             0, concat_sub['ieee_median']))
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_pushout_median.csv', 
                                        index=False, float_format='%.6f')


# # MinMax + Mean Stacking
# >* MinMax seems more gentle and it outperforms the previous one

# In[ ]:


concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['ieee_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['ieee_min'], 
                                             concat_sub['ieee_mean']))
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')


# # MinMax + Median Stacking

# In[ ]:


concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['ieee_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['ieee_min'], 
                                             concat_sub['ieee_median']))
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')


# # MinMax + BestBase Stacking
# >* loading submission with best score

# In[ ]:


sub_base = pd.read_csv('../input/ieeesubmissions/blend01.csv')


# In[ ]:


concat_sub['ieee_base'] = sub_base['isFraud']
concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['ieee_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['ieee_min'], 
                                             concat_sub['ieee_base']))
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_bestbase.csv', 
                                        index=False, float_format='%.6f')


# ## Median stacking gives the best LB score
