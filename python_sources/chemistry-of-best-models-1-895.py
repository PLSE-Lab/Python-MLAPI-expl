#!/usr/bin/env python
# coding: utf-8

# ## Stacking the Best Models
# <pre><b>This Kernel shows how the scores can be improved using Stacking Method.
# Credit Goes to the following kernels
# ref:
# 1. https://www.kaggle.com/filemide/distance-criskiev-hyparam-cont-1-662
# 2. https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
# 3. https://www.kaggle.com/marcelotamashiro/lgb-public-kernels-plus-more-features
# 4. https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
# 5. https://www.kaggle.com/fnands/1-mpnn/output?scriptVersionId=18233432
# 6. https://www.kaggle.com/harshit92/fork-from-kernel-1-481
# 7. https://www.kaggle.com/xwxw2929/keras-neural-net-and-distance-features
# 8. https://www.kaggle.com/marcogorelli/criskiev-s-distances-more-estimators-groupkfold?scriptVersionId=18843561
# 9. https://www.kaggle.com/toshik/schnet-starter-kit
# 10.https://www.kaggle.com/abazdyrev/nn-w-o-skew
# 
# </b></pre>

# ## Stat Stack

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


sub_path = "../input/chemistry-models"
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
concat_sub.iloc[:,1:].corr()


# In[ ]:


corr = concat_sub.iloc[:,1:].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap='prism', vmin=0.96, center=0, square=True, linewidths=1, annot=True, fmt='.4f')


# In[ ]:


# get the data fields ready for stacking
concat_sub['m_max'] = concat_sub.iloc[:, 1:].max(axis=1)
concat_sub['m_min'] = concat_sub.iloc[:, 1:].min(axis=1)
concat_sub['m_median'] = concat_sub.iloc[:, 1:].median(axis=1)


# In[ ]:


concat_sub.describe()


# In[ ]:


cutoff_lo = 0.8
cutoff_hi = 0.2


# # Mean Stacking

# In[ ]:


rank = np.tril(concat_sub.iloc[:,1:ncol].corr().values,-1)
m_gmean = 0
n = 8
while rank.max()>0:
    mx = np.unravel_index(rank.argmax(), rank.shape)
    m_gmean += n*(np.log(concat_sub.iloc[:, mx[0]+1]) + np.log(concat_sub.iloc[:, mx[1]+1]))/2
    rank[mx] = 0
    n += 1


# In[ ]:


concat_sub['m_mean'] = np.exp(m_gmean/(n-1)**2)


# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub['m_mean']
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_mean.csv', 
                                        index=False, float_format='%.6f')


# # Median Stacking

# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub['m_median']
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')


# # Pushout + Median Stacking
# >* Pushout strategy is bit aggresive

# In[ ]:


concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:7] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:7] < cutoff_hi, axis=1),
                                             0, concat_sub['m_median']))
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_pushout_median.csv', 
                                        index=False, float_format='%.6f')


# # MinMax + Mean Stacking
# >* MinMax seems more gentle and it outperforms the previous one

# In[ ]:


concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:7] > cutoff_lo, axis=1), 
                                    concat_sub['m_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:7] < cutoff_hi, axis=1),
                                             concat_sub['m_min'], 
                                             concat_sub['m_mean']))
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')


# # MinMax + Median Stacking

# In[ ]:


concat_sub['scalar_coupling_constant'] = np.where(np.all(concat_sub.iloc[:,1:7] > cutoff_lo, axis=1), 
                                    concat_sub['m_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:7] < cutoff_hi, axis=1),
                                             concat_sub['m_min'], 
                                             concat_sub['m_median']))
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')


# ## Averaging Ranks

# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub['mol0'].rank(method ='min') + concat_sub['mol1'].rank(method ='min') + concat_sub['mol2'].rank(method ='min') 
concat_sub['scalar_coupling_constant'] = (concat_sub['scalar_coupling_constant']-concat_sub['scalar_coupling_constant'].min())/(concat_sub['scalar_coupling_constant'].max() - concat_sub['scalar_coupling_constant'].min())
concat_sub.describe()
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_rank.csv', index=False, float_format='%.8f')


# Best Results : Stack Median
