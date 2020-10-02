#!/usr/bin/env python
# coding: utf-8

# * Thanks to
# >* https://www.kaggle.com/danmusetoiu/staking-and-stealing-like-a-molecule
# >* https://www.kaggle.com/marcelotamashiro/lgb-public-kernels-plus-more-features
# >* One of my submissions

# # [UpVote if this was helpful](http://)

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


sub1 = pd.read_csv('../input/mysubmissions/submission(-1.581).csv')
sample = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')
sub2 = pd.read_csv('../input/mysubmissions/submission(-1.587).csv')


# In[ ]:


sub1['scalar_coupling_constant'].describe()


# In[ ]:


sub2['scalar_coupling_constant'].describe()


# In[ ]:


sample['scalar_coupling_constant'] = sub2['scalar_coupling_constant'] 
sample.to_csv('stackers_blend.csv', index=False)


# In[ ]:


sns.distplot(sample['scalar_coupling_constant'])


# ## Upvote if this improved your score
