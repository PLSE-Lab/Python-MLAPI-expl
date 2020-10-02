#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


answer_argmax = []
for j in range(train.shape[0]):
    p = [0]*7
    a = [int(x) for x in train['visits'][j].split()]
    if a[-1] > 975:
        for i in a:
            p[(i-1)%7] += i
        answer_argmax.append(np.argmax(p) + 1)
    else:
        answer_argmax.append(0)


# In[ ]:


sub = pd.read_csv('../input/solutionex.csv')
sub['nextvisit'] = answer_argmax
sub.to_csv('solution.csv', index=False)

