#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[7]:


dfTrain = pd.read_csv("../input/train.csv")
dfTrain.head()


# In[8]:


dfTest = pd.read_csv("../input/test.csv")
dfTest.head()


# In[9]:


#Submission:
submissionDF = pd.DataFrame({"Id": dfTest["Id"],"Demand":0})
submissionDF.to_csv('Submissionv1.csv',index=False)

