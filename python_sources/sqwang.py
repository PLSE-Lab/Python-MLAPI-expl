#!/usr/bin/env python
# coding: utf-8

# thanks to [stacking?](https://www.kaggle.com/rajwardhanshinde/stacking)

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


outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "ieee" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


concat_sub.iloc[:,1:ncol].corr()


# In[ ]:


corr = concat_sub.iloc[:,1:7].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


concat_sub['ieee_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub['ieee_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub['ieee_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub['ieee_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)


# In[ ]:


concat_sub.describe()


# 

# In[ ]:


cutoff_lo = 0.90
cutoff_hi = 0.30


# In[ ]:


concat_sub['isFraud'] = concat_sub['ieee_median']
concat_sub[['TransactionID', 'isFraud']].to_csv('ieee.csv', 
                                        index=False, float_format='%.6f')

