#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle


# In[ ]:


test_ids = pd.read_csv("../input/ieee-sol1/test_ids.csv")


# In[ ]:


sub = pd.read_csv("../input/ieee-sol1/blend_of_blends_1.csv")


# In[ ]:


sub["uid"] = test_ids["uid3"].values


# In[ ]:


sub["isFraud"] = sub["uid"].map(sub.groupby("uid")["isFraud"].quantile(0.9))


# In[ ]:


del sub["uid"]


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




