#!/usr/bin/env python
# coding: utf-8

# # A big thanks to [roydatascience](https://www.kaggle.com/roydatascience)

# ## Stacking the Best Models
# <pre><b>
# This Kernel shows how the scores can be improved using Stacking Method.
# Credit Goes to the following kernels
# ref:
# 1. https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
# 2. https://www.kaggle.com/marcelotamashiro/lgb-public-kernels-plus-more-features
# 3. https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
# 4. https://www.kaggle.com/fnands/1-mpnn
# 5. https://www.kaggle.com/harshit92/fork-from-kernel-1-481
# 6. https://www.kaggle.com/marcogorelli/criskiev-s-distances-more-estimators-groupkfold?scriptVersionId=18843561
# </b></pre>

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


G = pd.read_csv('/kaggle/input/distance-is-all-you-need-hyper-search/submission.csv') #old dis hyp plb:-1.770


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
# concat_sub['mod_H'] = G['scalar_coupling_constant']
concat_sub.reset_index(inplace=True)

concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


concat_sub.head(5)


# In[ ]:



schnet  =  pd.read_csv("../input/schnet-starter-kit/kernel_schnet.csv")
concat_sub['new_mod_H'] = G.scalar_coupling_constant
concat_sub['schnet_mod_H'] = schnet.scalar_coupling_constant


# In[ ]:


concat_sub.head(6)


# In[ ]:


# check correlation
concat_sub.iloc[:,1:].corr()


# In[ ]:





# In[ ]:


corr = concat_sub.iloc[:,1:].corr()
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


concat_sub.drop('mol4', axis=1, inplace=True)


# In[ ]:


concat_sub.iloc[:,1:].corr()


# In[ ]:


concat_sub['m_median'] = concat_sub.iloc[:, 1:8].median(axis=1)


# # Median Stacking

# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub['m_median']
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')

