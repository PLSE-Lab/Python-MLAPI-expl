#!/usr/bin/env python
# coding: utf-8

# You can check how R,C and V score individually on LB in order to figure out where you lost most of your score. Be aware that R is twice as impactful than C and V.
# Most people will have lost a lot on C and R.
# 
# All you need to do is submit your solution with only predicting for example R and setting C and V to zero and then fill that number in the respective cell below. Same holds for C and V.
# We can come up with the equations as we know how a solution should score if we only predict zeros.

# In[ ]:


import pandas as pd
from sklearn import metrics
import numpy as np


# In[ ]:


train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
target_columns = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
y_train = train[target_columns].values


# In[ ]:


def metric(y, p):
    scores = []
    for i in range(3):
        y_true_subset = y[:,i]
        y_pred_subset = p[:,i]
        recalls = []
        for c in set(y_true_subset):
            idx = np.where(y_true_subset==c)
            s = (y_true_subset[idx] == y_pred_subset[idx]).mean()
            recalls.append(s)
        s = np.mean(recalls)
        scores.append(s)
    final_score = np.average(scores, weights=[2,1,1])
    return final_score, scores


# In[ ]:


r = np.zeros(len(train))
c = np.zeros(len(train))
v = np.zeros(len(train))
x = np.vstack([r,c,v]).T


# In[ ]:


metric(y_train, x)


# In[ ]:


_, scores = metric(y_train, x)


# In[ ]:


# calculate R score
r_lb = 0.5500
(0.25*scores[1] + 0.25*scores[2]) / (-0.5) + (r_lb / 0.5)


# In[ ]:


# calculate C score
c_lb = 0.2720
(0.5*scores[0] + 0.25*scores[2]) / (-0.25) + (c_lb / 0.25)


# In[ ]:


# calculate V score
v_lb = 0.2860
(0.25*scores[0] + 0.25*scores[1]) / (-0.25) + (v_lb / 0.25)


# In[ ]:




