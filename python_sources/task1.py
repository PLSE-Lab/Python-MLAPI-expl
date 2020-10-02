#!/usr/bin/env python
# coding: utf-8

# Task1 : 
# I will use the csv file of invoiced amounts to check invoiced amounts and to compared them with calculated accruals

# 

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/invoice/invoice-amounts.csv")


# **Data Exploration**

# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.columns.values


# ****Exploration of the file Datatype Dictionary

# In[ ]:


datatype = pd.read_csv("../input/datatype/Test 1 - FedEx Datatype Dictionary.csv")
datatype.shape

