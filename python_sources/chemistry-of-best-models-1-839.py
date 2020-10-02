#!/usr/bin/env python
# coding: utf-8

# This Kernel shows how the scores can be improved using Stacking Method.
# Credit Goes to the following kernels ref:
# 1. https://www.kaggle.com/filemide/distance-criskiev-hyparam-cont-1-662
# 2. https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
# 3. https://www.kaggle.com/marcelotamashiro/lgb-public-kernels-plus-more-features
# 4. https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
# 5. https://www.kaggle.com/fnands/1-mpnn/output?scriptVersionId=18233432
# 6. https://www.kaggle.com/harshit92/fork-from-kernel-1-481
# 7. https://www.kaggle.com/xwxw2929/keras-neural-net-and-distance-features
# 8. https://www.kaggle.com/marcogorelli/criskiev-s-distances-more-estimators-groupkfold?scriptVersionId=18843561
# 9. https://www.kaggle.com/toshik/schnet-starter-kit

# Also many thanks to:
# 
# 1. https://www.kaggle.com/filemide/chemistry-of-best-models-1-801
# 2. https://www.kaggle.com/roydatascience/chemistry-of-best-models-1-835

# Basically what I did here is just took median of outputs from two notebooks in the upper cell.
# 
# Moooooooore median stacking!!!

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import seaborn as sns
from subprocess import check_output


# In[ ]:


print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sub_path = "../input/stackingoutputs"
all_files = os.listdir(sub_path)
all_files


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in ['stack_median_1_801.csv', 'stack_median_1_822.csv']]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


concat_sub.head()


# In[ ]:


concat_sub['m_median'] = concat_sub.iloc[:, 1:].median(axis=1)


# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub['m_median']


# In[ ]:


concat_sub['m_median_1_835'] = pd.read_csv("../input/stackingoutputs/stack_median_1_835.csv")["scalar_coupling_constant"]


# In[ ]:


concat_sub['scalar_coupling_constant'] = concat_sub[["m_median", "m_median_1_835"]].median(axis=1)


# In[ ]:


concat_sub.head()


# In[ ]:


train = pd.read_csv("../input/champs-scalar-coupling/train.csv")


# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv("../input/champs-scalar-coupling/test.csv")


# In[ ]:


test.head()


# In[ ]:


concat_sub.info(verbose=True, null_counts=True)


# In[ ]:


test.info(verbose=True, null_counts=True)


# In[ ]:


concat_sub = pd.merge(concat_sub, test[['id', 'type']], on='id', how='left')


# In[ ]:


concat_sub.info(verbose=True, null_counts=True)


# In[ ]:


concat_sub.head()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
sns.boxplot(x="type", y="scalar_coupling_constant", data=train, order=train["type"].unique(), ax=ax1)
sns.boxplot(x="type", y="scalar_coupling_constant", data=concat_sub, order=train["type"].unique(), ax=ax2)
ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_median.csv', index=False, float_format='%.6f')


# In[ ]:




