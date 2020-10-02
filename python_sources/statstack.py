#!/usr/bin/env python
# coding: utf-8

# Thanks @DSEverything for https://www.kaggle.com/dongxu027/explore-stacking-lb-0-1463

# In[ ]:


import os
import numpy as np 
import pandas as pd 
from subprocess import check_output
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
print(check_output(["ls", "../input/"]).decode("utf8"))


# # Data Load

# In[ ]:


sub_path = "../input/champstacks"
all_files = os.listdir(sub_path)
all_files


# In[ ]:


# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "champ" + str(x), range(len(concat_sub.columns))))
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
concat_sub['champ_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub['champ_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub['champ_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub['champ_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)


# In[ ]:


concat_sub.describe()


# In[ ]:


cutoff_lo = -37
cutoff_hi = 205


# # Mean Stacking

# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub['champ_mean']
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_mean.csv', 
                                        index=False, float_format='%.6f')


# **LB----**

# # Median Stacking

# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub['champ_median']
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')


# ![](http://)**LB ----**

# # PushOut + Median Stacking 
# 
# Pushout strategy is a bit agressive given what it does...

# In[ ]:


concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             0, concat_sub['champ_median']))
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_pushout_median.csv', 
                                        index=False, float_format='%.6f')


# > **LB -----**

# # MinMax + Mean Stacking
# 
# MinMax seems more gentle and it outperforms the previous one given its peformance score.

# In[ ]:


concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['champ_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['champ_min'], 
                                             concat_sub['champ_mean']))
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')


# > **LB ----**
# 
# 

# # MinMax + Median Stacking 

# In[ ]:


concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['champ_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['champ_min'], 
                                             concat_sub['champ_median']))
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')


# **LB ----** -

# # MinMax + BestBase Stacking

# In[ ]:


# load the model with best base performance
sub_base = pd.read_csv('../input/champstacks/submission-giba-1 (2).csv')


# In[ ]:


concat_sub['champ_base'] = sub_base['scalar_coupling_constant']
concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['champ_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['champ_min'], 
                                             concat_sub['champ_base']))
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_minmax_bestbase.csv', 
                                        index=False, float_format='%.6f')


# > **LB----** -
