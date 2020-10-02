#!/usr/bin/env python
# coding: utf-8

# * Thanks to
# >* https://www.kaggle.com/danmusetoiu/staking-and-stealing-like-a-molecule
# >* https://www.kaggle.com/marcelotamashiro/lgb-public-kernels-plus-more-features

# # [UpVote if this was helpful](http://)

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


sub1 = pd.read_csv('../input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv')
sub2 = pd.read_csv('../input/staking-and-stealing-like-a-molecule/submission.csv')
sample = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')


# In[ ]:


sub1['scalar_coupling_constant'].describe()


# In[ ]:


sub2['scalar_coupling_constant'].describe()


# In[ ]:


sample['scalar_coupling_constant'] = (0.6*sub2['scalar_coupling_constant'] + 0.4*sub1['scalar_coupling_constant'])
sample.to_csv('stackers_blend.csv', index=False)


# In[ ]:


sns.distplot(sample['scalar_coupling_constant'])


# ## Upvote if this improved your score
