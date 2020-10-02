#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


animals = pd.read_csv("../input/train.csv")


# In[ ]:


# Defining has name method
def has_name(name):
    if name is np.nan:
        return "no"
    return "yes"


# In[ ]:


# Creating parameter HasName.
animals['HasName'] = animals.Name.apply(has_name)


# In[ ]:


HasName = animals['HasName'].value_counts() 
HasName.plot(kind='bar',color='#34ABD9',rot=0)


# In[ ]:


HasName = animals[['HasName','AnimalType']].groupby(['AnimalType','HasName']).size().unstack()
HasName.plot(kind='bar',color=['#34ABD8','#E98F85'],rot=-30)


# In[ ]:


HasName = animals[['HasName','OutcomeType']].groupby(['OutcomeType','HasName']).size().unstack()
HasName.plot(kind='bar',color=['#34ABD8','#E98F85'],rot=-30)


# In[ ]:


a = animals[['HasName','AnimalType','OutcomeType']].groupby(['OutcomeType','AnimalType','HasName']).size().unstack().unstack()
a.plot(kind='bar',stacked=False,figsize=(10,8),rot=-30)

