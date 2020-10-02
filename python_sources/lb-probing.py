#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# # Basic idea

# Here is a bit of exploitation/exploration ;)
# 
# I made submissions to check what is the distribution of each class in the public test set. 
# Please bear in mind that the private test set might be DIFFERENT (it has been different in almost all real competitions I've observed/paticipated). There is no guarantee whatsoever. 
# So the main goal of this competition is still to come up with the solution capable to generalize well.

# In[ ]:


classes = {'Cover_type1' : 0.37053,'Cover_type2' : 0.49657,'Cover_type3' : 0.059647,'Cover_type4' : 0.00106,'Cover_type5' : 0.01287,'Cover_type6' : 0.02698,'Cover_type7' : 0.03238}
pd_cl = pd.DataFrame.from_dict(classes, orient='index', columns=['Share'])


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(x = pd_cl.index, y = pd_cl.Share  *100,  data = pd_cl, ax = ax)


# This type of distribution is essentially what one would want to achieve to get the perfect score.
# We see that classes 1 and 2 give us 86.6% of all data in the public test set. Therefore predicting them alone perfectly will give us appropriate LB score. 
# Is it doable though?

# # Verdict

# From what I've tried so far (LGBM, MLP, RandomForest), the distribution of the predicted labels is far from perfect.
# I'm planning on feature engeneering. Also will be looking to tune my NN model with grid-search.

# I'll update this notebook as soon as I get some ideas about the close matching predictions from various models.
