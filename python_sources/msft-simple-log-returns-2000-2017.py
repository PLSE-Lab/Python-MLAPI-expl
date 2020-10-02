#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_datareader import data as wb
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt


# First, I am loading in the data and previewing what it looks like.
# For any first-time users, you can find the link to pull in your csv file within workspace section. Use pandas package to read the csv file. I also indexed the date column for convenience. 

# In[23]:


MSFT = pd.read_csv("../input/MSFT_2000.csv", index_col = 'Date')
MSFT.head()


# Now, I will go ahead and calculate the daily price % changes. Remember (P1/P0)-1 = daily price change:

# In[24]:


MSFT["simple_return"] = (MSFT["MSFT"] / MSFT["MSFT"].shift(1)) - 1
print (MSFT["simple_return"])


# Now I will plot what the daily return fluctuations look like using matplotlib:

# In[25]:


MSFT['simple_return'].plot(figsize=(15, 10))
plt.show()


# Average daily and yearly returns? Coming right up.
# There are about 250 days where the stock market is open in a 365 year, that's why I multilpy the average daily return by that number to find the yearly value:

# In[26]:


daily_return = MSFT['simple_return'].mean()
print ("Daily Return Average is: "+str(round(daily_return,4)*100)+ "%")

yearly_return = MSFT['simple_return'].mean()*250
print ("Yearly Return Average is: "+str(round(yearly_return,4)*100)+ "%")


# Now, let's start looking into the logarithmic returns for this stock. We're going to use numpy for this calculation. ln(P1/P0) is the formula for this. Format is almost identical to the simple return above.  

# In[27]:


MSFT["log_return"] = np.log(MSFT["MSFT"]/MSFT["MSFT"].shift(1))
print (MSFT["log_return"])


# Results:

# In[28]:


daily_return_log = MSFT['log_return'].mean()
print ("Daily Log Return Average is: "+str(round(daily_return_log,4)*100)+ "%")

yearly_return_log = MSFT['log_return'].mean()*250
print ("Yearly Log Return Average is: "+str(round(yearly_return_log,4)*100)+ "%")

