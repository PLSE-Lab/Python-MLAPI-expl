#!/usr/bin/env python
# coding: utf-8

# Credit to origin author for MinMaxBestBaseStacking (https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/utils.py)
# 
# 

# 1. Data from https://www.kaggle.com/truonghoang/siimisic-submission-files with max score: 0.868
# 
#     I take 11 top submission files for this notebook and apply:
# >     * median           : 0.878
# >     * mean             : 0.867
# >     * minmax_mean      : 0.860
# >     * pushout_median   : 0.819
# 
# 2. Data from https://www.kaggle.com/truonghoang/siimisic-submission-files-cpu with max score: 0.884
#     
#     I take 10 top submission files for this notebook and apply:
# >     * median           : 0.893
# >     * mean             : 0.891
# >     * minmax_mean      : 0.888
# 
# 3. Data from https://www.kaggle.com/truonghoang/multi-size-eff-lb-0-912 with max score: 0.912
# 
#     I take 5 submission files for this notebook and apply:
# >     * median           : 0.914

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


sub_path = "../input/top-my-submission"
all_files = os.listdir(sub_path)


# In[ ]:


outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files if '.csv' in f and '_me' in f]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


# get the data fields ready for stacking
concat_sub['target_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub['target_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)


# In[ ]:


concat_sub.describe()


# In[ ]:


concat_sub['target'] = concat_sub['target_mean']
concat_sub[['image_name', 'target']].to_csv('submission_mean.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['target'] = concat_sub['target_median']
concat_sub[['image_name', 'target']].to_csv('submission_median.csv', 
                                        index=False, float_format='%.6f')

