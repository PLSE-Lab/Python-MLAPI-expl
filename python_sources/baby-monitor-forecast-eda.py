#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.head()


# In[ ]:




