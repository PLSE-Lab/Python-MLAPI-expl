#!/usr/bin/env python
# coding: utf-8

# Hi. I'm new here and will try to explore this Google Play Store data set.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# Read data to csv and check head of the data

# In[ ]:


goog_play = pd.read_csv('../input/googleplaystore.csv')
goog_play.head()


# Check dataytypes and missing values

# In[ ]:


goog_play.info()


# In[ ]:


goog_play.describe()


# Maximum rating is 19? Something wrong?

# In[ ]:


goog_play[goog_play['Rating'] == 19]


# Compare with others

# In[ ]:


goog_play[10470:10475]


# The catagory data of this app "Life Made WI-Fi Touchscreen Photo Frame" is missing and it needs one shift to the right..

# In[ ]:


goog_play.iloc[10472,1:] = goog_play.iloc[10472,1:].shift(1)
goog_play[10470:10475]


# Crosscheck data with Play Store to find its category.. yep its rating is really 1.9
# 
# ![Life%20Made%20WI-Fi%20Touchscreen%20Photo%20Frame.jpg](attachment:Life%20Made%20WI-Fi%20Touchscreen%20Photo%20Frame.jpg)

# In[ ]:


goog_play.iloc[10472,1] = 'LIFESTYLE'
goog_play[10470:10475]


# In[ ]:


goog_play.describe()


# In[ ]:


goog_play.dtypes


# All types are object now. Convert 'Rating' and 'Reviews' to float.

# In[ ]:


goog_play['Rating'] = goog_play['Rating'].apply(pd.to_numeric, errors='coerce')
goog_play['Reviews'] = goog_play['Reviews'].apply(pd.to_numeric, errors='coerce')


# In[ ]:


goog_play.dtypes


# Plot histograms of 'Rating' and 'Reviews'

# In[ ]:


#Histogram
goog_play["Rating"].plot(kind="hist",color="blue",bins=30,grid=True,alpha=0.65,label="Rating",figsize=(10,10))
plt.legend()
plt.xlabel("Rating")
plt.title("Rating Distribution")
plt.show()


# In[ ]:


#Histogram
goog_play["Reviews"].plot(kind="hist",color="blue",bins=30,grid=True,alpha=0.65,label="Reviews",figsize=(10,10))
plt.legend()
plt.xlabel("Reviews")
plt.title("Reviews Distribution")
plt.show()


# In[ ]:




