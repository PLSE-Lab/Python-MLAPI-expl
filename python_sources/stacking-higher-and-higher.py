#!/usr/bin/env python
# coding: utf-8

# [Fork of 'Stacking?'](https://www.kaggle.com/rajwardhanshinde/stacking)
# 
# **Update for v.9:** again a slightly higher scoring Under_Over submission is added (nothing dramatic: 0.9398). You may have to remove and reattach ieeesubmissions5 dataset to get the latest version (`xgboost_under_over_blend_9398`)
# 
# **Update for v.8:** a tiny update with a fresh file from [Undersample with multiple runs kernel](https://www.kaggle.com/stocks/under-sample-with-multiple-runs). don't know the score yet
# 
# **Update for v.7**:
# 
# Added [IEEE - LGB + Bayesian opt.](https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt)
# 
# **Update for v.5**:
# 
#  * <font color=green>Median 0.9425</font>
#  * <font color=green>Mean 0.9420 </font>
#  * <font color=green>Median Rank 0.9413</font>
#  * <font color=green>Mean Rank 0.9413 </font>
# 
# * `ieeesubmissions2` dataset was update replacing `safebox9416.csv` with `safebox9367.csv`. `safebox9416.csv` had an incorrect score linked to it (it is actually 0.9322). The reason for the mix-up is that it is unknown which submission generates which score in public kernels. The reason I am no longer using `blend1.csv` from Safebox kernel is because `blend1` itself was a blend with `ieee_9383` also used in this kernel.
# 
# * added Median rank
# 
# **Update for v.4**:
# 
# v.4 is the same as v.3, only reporting select scores for v.3
# 
# **Update for v.3**:
# 
# **<font color=green>Submissions tried in v.3:</font>**
# 
#  * <font color=green>Median 0.9427</font>
#  * <font color=green>Mean 0.9420 </font>
# 
# I added a new dataset with submission files labelled by their scores (to the best of my knowledge). They all come from the same kernels as in v.2 but two are different  versions of those. Not sure if any them might improve the score.
# 
# **Updates made by me in v.1 and v.2:**
# 
# * use the higher scoring blend of oversample + undersample from [My kernel](https://www.kaggle.com/stocks/under-sample-with-multiple-runs). 
# I'm guessing that the stacking kernel that I've copied here used only my under-sampled model.
# At least one of the other models [EDA Kernel](https://www.kaggle.com/artgor/eda-and-models) is scoring higher.
# Thus, the explanation for high score of this clone likely lies in higher scoring models used for stacking.
# * trying stacking based on ranks
# 
# **<font color=green>Submission tried in v.2:</font>**
# 
#  * <font color=green>Median 0.9429</font>
#  * <font color=green>Mean rank 0.9415 </font>
# 
# #### Info from the parent of this clone:
# >* Based on https://www.kaggle.com/lpachuong/statstack
# >* Thanks to <br>
# https://www.kaggle.com/jazivxt/safe-box<br>
# https://www.kaggle.com/artgor/eda-and-models<br>
# https://www.kaggle.com/stocks/under-sample-with-multiple-runs<br>
# https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb

# ## <font color=blue>Vote early and vote often!</font>
# 

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


sub_path = "../input/ieeesubmissions5"
#sub_path = "../input/ieeesubmissions3"  # Uncomment to get v.7 output
#sub_path = "../input/ieeesubmissions2" # Uncomment this line to access the input files that should match v.5 and .6 outputs
#sub_path = "../input/ieeesubmissions"  # Uncomment this line to access the input files that should match v.2 output
all_files = os.listdir(sub_path)
all_files
all_files.remove('blend01.csv')
all_files.remove('blend02.csv')
all_files


# In[ ]:


outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "ieee" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
ncol = concat_sub.shape[1]
concat_sub.head()


# In[ ]:


concat_sub_rank = concat_sub.iloc[:,1:ncol].copy()
concat_sub_rank.head()
for _ in range(ncol-1):
    concat_sub_rank.iloc[:,_] = concat_sub_rank.iloc[:,_].rank(method ='average')
concat_sub_rank.describe()
    


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


# # Median Rank

# In[ ]:


concat_sub['isFraud'] = concat_sub_rank.median(axis=1)
concat_sub['isFraud'] = (concat_sub['isFraud']-concat_sub['isFraud'].min())/(concat_sub['isFraud'].max() - concat_sub['isFraud'].min())
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_median_rank.csv', index=False, float_format='%.8f')


# # Mean Rank

# In[ ]:


concat_sub['isFraud'] = concat_sub_rank.mean(axis=1)
concat_sub['isFraud'] = (concat_sub['isFraud']-concat_sub['isFraud'].min())/(concat_sub['isFraud'].max() - concat_sub['isFraud'].min())
concat_sub[['TransactionID', 'isFraud']].to_csv('stack_mean_rank.csv', index=False, float_format='%.8f')

