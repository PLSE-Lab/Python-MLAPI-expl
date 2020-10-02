#!/usr/bin/env python
# coding: utf-8

# # Scraping Saudi Hotels Information from Booking.com 

# ## Problem Statement 
# 
# It's always hard to choose the perfect hotel when travling as we have to take into consideration many diffrent aspects so we usually take the easier path of looking at the rating and the reviews of each hotel. Here I collected data for hotels in the three biggest citys in Saudi Arabia(Riyadh, Jeddah and  and the Eastern Province) also Makkah and Madina since they are two of the biggest islamic destination for muslims to visit.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import pandas as pd


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/Saudi_Hotels.csv')


# ### Data Description
#     
# |Feature|Type|Description|
# |---|---|---|
# |<div align="center">HotelName</div>|object|<div align="left">The name of the hotel (it could be in English or Arabic or both)</div>| 
# |<div align="center">City|object</div>|<div align="left">The city where the hotel is</div>| 
# |<div align="center">DistenceFromCenter</div>|float|<div align="left">A column that tells us how far is the hotel from the city center in miles</div>| 
# |<div align="center">Description</div>|object|<div align="left">The written description of the hotel in the site booking.com</div>| 
# |<div align="center">NumberOfReviews</div>|int|<div align="left">Number of reviews written about the hotel</div>| 
# |<div align="center">Grade|object</div>|<div align="left">The grade of the hotel given by reviewres (Exceptional, Awesome, Excellent, Very Good, Good, Below Good)</div>| 
# |<div align="center">Rating</div>|float|<div align="left">The rating of the hotel given by reviewrs</div>|

# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


ax = df['City'].value_counts().plot(kind = 'barh', title = 'Number of hotels from each city')
ax.set_ylabel('City')


# In[ ]:


df.groupby('City')['Rating'].mean().plot(kind = 'barh', title = 'Avarage rating for each city')


# ## Conclusion
# 
# I collected this data with the follwing questions in mind and the thought that this data might answer them:
# - Does the location of the hotel affect the ratings of the customers?
# - What kind of key words does high graded hotels have in their descriptions?
# - Overall do certain cities have better hotels?
