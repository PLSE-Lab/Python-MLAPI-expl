#!/usr/bin/env python
# coding: utf-8

# ## The submission files of this notebook is from [This notebook](https://www.kaggle.com/gc1023/ensemble-top-kernels) which has LB score of 98.1. 
# 
# ## I tried to play with entropy. My idea was to select the probability values from based on their entropy. I tried with maximum and minimum entropy values. I got the same public score of 98.1 by selecting probability distribution with minimum entropy. What do you think of this?  
# 
# ## Since I am new to kaggle, your valuable opinion will be highly appreciated. 
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from scipy.stats import entropy


# # Data Loading

# In[ ]:


sub1 = pd.read_csv('../input/average-efficientnet/submission.csv')
sub2 = pd.read_csv('../input/classification-densenet201-efficientnetb7/submission.csv')
sub3 = pd.read_csv('../input/tf-zoo-models-on-tpu/submission.csv')
sub4 = pd.read_csv('../input/fork-of-plant-2020-tpu-915e9c/submission.csv')
sub = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')


# # Calculate Entropies

# In[ ]:


ent1 = entropy(sub1.loc[:,'healthy' : ], base=2, axis = 1)
ent2 = entropy(sub2.loc[:,'healthy' : ], base=2, axis = 1)
ent3 = entropy(sub3.loc[:,'healthy' : ], base=2, axis = 1)
ent4 = entropy(sub4.loc[:,'healthy' : ], base=2, axis = 1)
entropies = np.array([ent1, ent2, ent3, ent4]).transpose()
entropies.shape

selected = np.argmin(entropies, axis = 1)


# In[ ]:


# pred1 = np.argmax(np.array(sub1.loc[:,'healthy' : ]), axis = 1)
# pred2 = np.argmax(np.array(sub2.loc[:,'healthy' : ]), axis = 1)
# pred3 = np.argmax(np.array(sub3.loc[:,'healthy' : ]), axis = 1)
# pred4 = np.argmax(np.array(sub4.loc[:,'healthy' : ]), axis = 1)
# preds = np.array([pred1, pred2, pred3, pred4], dtype=np.int32).transpose()


# In[ ]:


# for i in range(1821):
#     if len(set(preds[1]))>1:
#         selected = selected_argmax[i]
#         if selected ==0:
#             sub.loc[i, 'healthy' : ] = sub1.loc[i, 'healthy' :]
#         elif selected ==1:
#             sub.loc[i, 'healthy' : ] = sub2.loc[i, 'healthy' :]
#         elif selected == 2:
#             sub.loc[i, 'healthy' : ] = sub3.loc[i, 'healthy' :]
#         elif selected == 3:
#             sub.loc[i, 'healthy' : ] = sub4.loc[i, 'healthy' :]
    
#     else:
#         selected = selected_argmin[i]
#         if selected ==0:
#             sub.loc[i, 'healthy' : ] = sub1.loc[i, 'healthy' :]
#         elif selected ==1:
#             sub.loc[i, 'healthy' : ] = sub2.loc[i, 'healthy' :]
#         elif selected == 2:
#             sub.loc[i, 'healthy' : ] = sub3.loc[i, 'healthy' :]
#         elif selected == 3:
#             sub.loc[i, 'healthy' : ] = sub4.loc[i, 'healthy' :]
        
        
        


# In[ ]:


# sub.healthy = ( sub1.healthy + sub2.healthy + sub3.healthy + sub4.healthy)/4
# sub.multiple_diseases = (sub1.multiple_diseases + sub2.multiple_diseases + sub3.multiple_diseases + sub4.multiple_diseases)/4
# sub.rust = (sub1.rust + sub2.rust + sub3.rust + sub4.rust)/4
# sub.scab = (sub1.scab + sub2.scab + sub3.scab + sub4.scab)/4


# In[ ]:


submission_size = len(selected)
for i in range(submission_size):
    if selected[i] ==0:
        sub.loc[i, 'healthy' : ] = sub1.loc[i, 'healthy' :]
    elif selected[i] ==1:
        sub.loc[i, 'healthy' : ] = sub2.loc[i, 'healthy' :]
    elif selected[i] == 2:
        sub.loc[i, 'healthy' : ] = sub3.loc[i, 'healthy' :]
    elif selected[i] == 3:
        sub.loc[i, 'healthy' : ] = sub4.loc[i, 'healthy' :]


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




