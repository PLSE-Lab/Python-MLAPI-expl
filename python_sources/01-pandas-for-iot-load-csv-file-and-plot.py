#!/usr/bin/env python
# coding: utf-8

# # Initial Data Analysis Notebook
# 
# - by Clayton Miller
# - Feb 23, 2018
# 
# In this notebook, I will load a raw data file and analyse the meter data inside.
# 
# First thing to do is import the pandas library!

# In[ ]:


import pandas as pd


# In[ ]:


Abigail_Data = pd.read_csv("../input/Office_Abigail.csv")


# In[ ]:


Abigail_Data.head()


# Instead of having the index be `0-8760`, let's make the timestamp the index!

# In[ ]:


Abigail_Data = pd.read_csv("../input/Office_Abigail.csv", index_col='timestamp')


# In[ ]:


Abigail_Data.head()


# In[ ]:


Abigail_Data.info()


# In[ ]:


Abigail_Data = pd.read_csv("../input/Office_Abigail.csv", index_col='timestamp', parse_dates=True)


# In[ ]:


Abigail_Data.info()


# In[ ]:


Abigail_Data.plot()


# In[ ]:


Abigail_Data.plot(figsize=(20,10))


# In[ ]:


Abigail_Data.info()


# ## Convert from Hourly Data to Daily Data
# 

# In[ ]:


Abigail_Data_Daily = Abigail_Data.resample("D").sum()


# In[ ]:


Abigail_Data_Daily.head()


# In[ ]:


Abigail_Data_Daily.info()


# In[ ]:


Abigail_Data_Daily.plot(figsize=(20,10))


# In[ ]:




