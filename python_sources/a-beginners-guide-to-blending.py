#!/usr/bin/env python
# coding: utf-8

# Here I give beginners a lesson about the proper way to create a blend to get a high score.  I see a lot of kernels that are actually blends of blends, and this is for the most part not a good idea.  
# 
# A blended solution works best if you **use models that are diverse** as well as give a good score.  So for example, don't blend 2 solutions that both use the same features and the same modeling technique.  A good way to start is to go to the kernels, sort them by best score, then look at each one and find the 2 or 3 top kernels that are not blends themselves.  Preferably choose solutions that use different features and a different modeling approach.  
# 
# I did that, and what I see are these 3 amazing kernels:
# 
# Distance - is all you need.
# by Sergii
# https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
# 
# LGB public kernels plus more features
# by marcelotamashiro 
# https://www.kaggle.com/marcelotamashiro/lgb-public-kernels-plus-more-features
# 
# MPNN
# by fnands
# https://www.kaggle.com/fnands/1-mpnn
# 
# Please go and give each of these kernels an upvote.  
# 
# The the next step is to blend their solutions together using a weighted average approach.  There is some trial and error here to determine the best weights to use for each solution.  We have 5 submissions per day here so you have a lot of tries to get the best weights possible.
# 
# Once you download the solutions provided by other folks, the code is trivial.

# In[ ]:


import pandas as pd

one = pd.read_csv('../input/champs-blending-tutorial/1.csv')
two = pd.read_csv('../input/champs-blending-tutorial/2.csv')
three = pd.read_csv('../input/blendsub/submission4.csv')

submission = pd.DataFrame()
submission['id'] = one.id
submission['scalar_coupling_constant'] = (0.37*one.scalar_coupling_constant) + (0.37*two.scalar_coupling_constant) + (0.26*three.scalar_coupling_constant)

submission.to_csv('super_blend3.csv', index=False)


# Then just submit and bask in the glory of these other people's work.
