#!/usr/bin/env python
# coding: utf-8

# ## Minh Vo

# ## Visualize forex price using custom charts
# 

# ## Import the required modules

# In[ ]:


import datetime
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Read in Currency data from https://www.investing.com

# In[ ]:


forex = pd.read_csv("../input/data-euro-usd.csv")


# ## Explore the data -- columns, data types and if there's any null data

# In[ ]:


forex.info()


# ## Show some records

# In[ ]:


forex.head(5)


# ## Need to convert date to datetime format

# In[ ]:


forex['Date'] = pd.to_datetime(forex['Date'])


# ## Set the index to be the new date

# In[ ]:


forex.index = forex['Date']


# ## Delete the original Date column (value), because we have set the index to be the new Date

# In[ ]:


del forex['Date']


# ## Display the rows, now that we've manipulated the index and Date

# In[ ]:


forex.head(5)


# ## Plot Price as a line graph

# In[ ]:


plt.figure(figsize=(12, 6))
forex['Price'].plot()
plt.show()


# ## Plot % Change Graph to see the Euro to USD forex price

# In[ ]:


plt.figure(figsize=(12, 6))
forex['Change %'].plot()
plt.show()


# ## Plotting 50 day moving average

# In[ ]:


forex['Moving Average 50 Days'] = forex.Price.rolling(window=50).mean()




# ## Add data to forex dataframe as new column (moving avg 50 days) for 1yr

# In[ ]:


data = forex[['Price', 'Moving Average 50 Days']][-365:]


# ## Show both graphs on one plot by setting subplots=False

# In[ ]:


plots = data.plot(subplots=False,figsize=(12, 5))
plt.show()


# ## Plot moving average 100 days to compare against 50 day plot

# In[ ]:


forex['Moving Average 100 Days'] = forex.Price.rolling(window=100).mean()


# ## Show plots separately, by setting subplots=True
# 

# In[ ]:


data = forex[['Price', 'Moving Average 100 Days']][-365:]
plots = data.plot(subplots=True,figsize=(12, 5))
plt.show()

