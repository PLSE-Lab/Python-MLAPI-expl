#!/usr/bin/env python
# coding: utf-8

# ### <h3> Zomato Restaurant Profiling

# <h4> Business Problem : </h4>
# 
# - 'The Roastery Coffee House' is one of the restaurant that is associated with Zomato , the owner of the restaurant has approached Zomato requesting them to analyze the performance of the restaurant and provide some insights.
# - Zomato has given you some data and asked you to carry out the task.
# - The objective is to analyze the data and present the insights to the restaurant owner , it should be presented in the form of a static dashboard

# ---

# Import Libraries

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


# ---

# Read Data

# In[ ]:


df = pd.read_csv('/kaggle/input/dataset/ZomatoProfiling.csv')
df.head()


# DATA DESCRIPTION <br> <br>
# date : date<br>
# rating : rating by user <br>
# res_id : restaurant id <br>
# res_name : restaurant name<br>
# rev_count : number of reviews given by that reviewer and number of followers of that reviewer<br>
# rev_id L reviewer id<br>
# rev_name : reviewer name<br>
# text : review text

# ---

# <h5> Extract only the data of  'The Roastery Coffee House' from the given data

# In[ ]:


df = df[df['res_name']=='The Roastery Coffee House']
df.head(2)


# In[ ]:


df.shape


# <h5> We now have 167 records and 8 columns

# <h5> Missing Ratio

# In[ ]:


mr = (df.isna().sum()/len(df))*100
pd.DataFrame(mr.sort_values(ascending=False) , columns=['Missing Ratio'])


# In[ ]:


df.fillna('NA',inplace=True)


# ---

# <h4> Exploratory Analysis

# ---

# <h5> Granularity : </h5>
# 
#   - Each record is a review and rating given to the restaurant by a particular user at a given time
#   
# <h5> Data Types : </h5>
# 
#   - The types of data we have :
#       - Numaric , DateTime & Text 
#  
# <h5> Business Use-Case: </h5>
# 
#   - The restaurant 'The Roastery Coffee House's profiling
# 
# <h5> Types of Analysis that can be done : </h5>
# 
#    - Distribution Analysis
#    - Text Analysis
#    - Trend Analysis

# ---

# <h4> Data Cleaning

# ---

# Extract Date values

# In[ ]:


df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['time'] = df['date'].dt.time


# Split rating

# In[ ]:


df['rating'] = df['rating'].apply(lambda x: x.split(' ')[1])


# Split user-reviews and  user-followers

# In[ ]:


df['user_reviews'] = df['rev_count'].apply(lambda x: x.split(',')[0])
df['user_reviews'] = df['user_reviews'].apply(lambda x: x.split(' ')[0])


# In[ ]:


df['user_followers']= df['rev_count'][df['rev_count']!='NA'].apply(lambda x: x.split(',')[1])
df['user_followers'] = df['user_followers'].fillna('NA')

df['user_followers'] = df['user_followers'][df['user_followers']!='NA'].apply(lambda x: x.split(' ')[1])


# In[ ]:


df.drop('rev_count',1,inplace=True)


# In[ ]:


df['rating'] = df['rating'].astype(float)
df['user_reviews'][df['user_reviews']=='NA'] = np.NaN
df['user_reviews'].fillna(0 ,inplace=True)
df['user_reviews'] = df['user_reviews'].astype(int)


# In[ ]:


df.head()


# ---

# <h4> Restaurant Profiling - The Roastery Coffee House

# ---

# In[ ]:


df.head(2)


# In[ ]:


sns.barplot(df['month'][df['year']==2019] , df['user_reviews'] )
plt.xlabel('Month')
plt.ylabel('Total Reviews')
plt.title('Total Reviews in 2019')
plt.xticks([0,1,2],labels=['January','Feburary','March'])
plt.show()


# In[ ]:


sns.barplot(df['month'][df['year']==2019] , df['rating'] )
plt.xlabel('Month')
plt.ylabel('Average Rating')
plt.title('Average Rating in 2019')
plt.xticks([0,1,2],labels=['January','Feburary','March'])
plt.show()


# <h5> Total 5-Star Ratings

# In[ ]:


df[df['rating']==5.0].count()[1]


# <h5> Ratings below 4.0

# In[ ]:


df[df['rating']<4].count()[1]


# In[ ]:


df.resample('1m',on='date').size().plot()
plt.grid(True)
plt.title('Reviews per Month')
plt.xlabel('Month')
plt.ylabel('Reviews')


# In[ ]:


df['text'][df['text']=='NA'] = np.NaN


# In[ ]:


from wordcloud import WordCloud
ip_string=' '.join(df['text'].str.replace('RATED',' ').dropna().to_list())

wc=WordCloud(background_color='white').generate(ip_string.lower())
plt.imshow(wc)
plt.show()


# ---

# ## DashBoard

# In[ ]:


from IPython.display import Image 
Image("/kaggle/input/dashboard/ZomatoRestaurantProfiling.JPG")


# ---
