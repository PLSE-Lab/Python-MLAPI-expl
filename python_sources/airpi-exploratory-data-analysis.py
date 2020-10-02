#!/usr/bin/env python
# coding: utf-8

# # Exploratory analysis of the AirPi data set
# 
# In this notebook we'll take a quick look at the AirPi data set. After loading the libraries needed for data processing and visualization we are going to:
# 1.  Look at 1D histograms of the individual histograms to check our data quality
# 2. Clean-up any data issues we spot
# 3. Create some 2D scatterplots to look for correlations between variables
# 
# ## Setting things up

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # statistical data visualization
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


# Now load the data file and check to see if it holds the data we're expecting.

# In[8]:


airpi = pd.read_csv("../input/AirPi Data - AirPi.csv")
airpi.head()


# ## Step 1: Look at individual series
# 
# Look at 1D histograms of all variables to get a sense of the data quality. With pandas, we only need one line of code for this!

# In[54]:


airpi.hist(color='red', figsize=(15, 9), bins=50)


# ## Step 2: Cleaning up the data
# Most data actually looks pretty good, except for the Temperature-DHT sensor and the Relative_Humidity sensor.
# 
# * We know that the Temperature-DHT in degrees Celsius must be larger than 0. My home office might not be the hottest place on earth, it's definitely not freezing!
# * We also know that Relative_Humidity must be between 0 and 100 percent.
# 
# So let's go ahead and remove all events where these conditions are not met and plot our 1D histograms again.
# 

# In[53]:


airpi.drop(airpi[airpi['Temperature-DHT [Celsius]'] < 0].index, inplace=True)
airpi.drop(airpi[airpi['Relative_Humidity [%]'] < 0].index, inplace=True)
airpi.drop(airpi[airpi['Relative_Humidity [%]'] > 100].index, inplace=True)
airpi.hist(color='green', figsize=(15, 9), bins=50)


# Much better! Time to start hunting for correlations, and hopefully some insights.
# 
# ## Step 3:  Create 2D scatterplots
# This we do using the cleaned-up data set from the last step. Again pandas enables us to create an inital matrix of scatterplots with one command.

# In[59]:


from pandas.plotting import scatter_matrix
scatter_matrix(airpi, alpha=0.4, figsize=(18, 18), diagonal='kde')


# In[ ]:




