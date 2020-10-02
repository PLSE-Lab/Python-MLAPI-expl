#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant reviews.csv')

df = pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')


# In[ ]:


data.shape, df.shape


# In[ ]:


data.head()


# In[ ]:


df.head()


# In[ ]:


data.isna().sum()


# In[ ]:


df.isna().sum()


# In[ ]:


data = data.dropna()


# In[ ]:


data.isna().sum()


# In[ ]:


df = df.dropna()


# In[ ]:


df.isna().sum()


# In[ ]:


data.shape, df.shape


# In[ ]:


plt.style.use('fivethirtyeight')
sns.countplot(data=data,x='Rating')


# In[ ]:


data['Rating'] = data['Rating'].str.replace('Like','5')


# In[ ]:


sns.countplot(data=data,x='Rating')


# In[ ]:


data['Rating'] = data['Rating'].astype(float) 


# In[ ]:


data.head()


# In[ ]:


data['Time'] = pd.to_datetime(data['Time'])
data['Time'].min(),data['Time'].max()


# In[ ]:


data.head()


# In[ ]:


data['Metadata'] =data['Metadata'].str.replace(' Review', ' Reviews')


# In[ ]:


data.head()


# In[ ]:


data['reviews'] = data['Metadata'].str.replace('[^0-9,]','').str.split(',').str[0].astype(float)
data['followers'] = data['Metadata'].str.replace('[^0-9,]','').str.split(',').str[1].astype(float)


# In[ ]:


data.head()


# In[ ]:


data['reviews'] = data['reviews'].astype(float)


# In[ ]:


data['followers'].fillna('0', inplace = True)


# In[ ]:


data['followers'] = data['followers'].astype(float)


# In[ ]:


data['Time'] = pd.to_datetime(data['Time'])
data['Day'] = data['Time'].dt.day
data['Month'] = data['Time'].dt.month
data['Year'] = data['Time'].dt.year


# In[ ]:


data.head()


# In[ ]:


x = data.groupby(['Restaurant','Rating'])['Rating'].count()
y = x.sort_values(ascending=False).head(11)
y


# In[ ]:


plt.figure(figsize=(15, 8))
res_rating_5 = data.groupby(['Restaurant','Rating'])['Rating'].count()
top_res_having_5_ratings = res_rating_5.sort_values(ascending = False).head(11)
chart1 = top_res_having_5_ratings[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Number_of_5_Ratings')
plt.xlabel('Restaurant_Name')


# In[ ]:


data['Pictures'].unique()


# In[ ]:


plt.figure(figsize=(15,8))
res_pic_20 = data.groupby('Restaurant')['Pictures'].max()
sort_pic_values = res_pic_20.sort_values(ascending=False).head(21)
 # end to beginning, counting down by 1
charts = sort_pic_values[::-1].plot.bar()
for p in charts.patches:
    charts.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Number_of_Pictures_taken_in_restaurants')
plt.xlabel('Restaurant_Name')


# In[ ]:


plt.style.use('dark_background')
plt.figure(figsize=(10,7))
top_10_res = data.groupby('Restaurant')['Rating'].mean()
top_10_res_ratings = top_10_res.sort_values(ascending=False).head(10)
chart = top_10_res_ratings[::-1].plot.bar()
for p in chart.patches:
    chart.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Ratings of Restuarants')
plt.xlabel('Restaurant_Name')


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(15,8))
top_10_reviewrs = data.groupby('Reviewer')['followers'].sum()
top_10_rev_followers = top_10_reviewrs.sort_values(ascending=False).head(10)
chart1 = top_10_rev_followers[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Number of Followers')
plt.xlabel('Reviewer Name')


# In[ ]:


plt.figure(figsize=(15, 4))
res_avg_rating = data.groupby(['Restaurant', 'Year'])['Rating'].mean()
top10_res = res_avg_rating.sort_values(ascending = False).head(10)
chart2 = top10_res[::-1].plot.bar()
for p in chart2.patches:
    chart2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Rating')
plt.xlabel('Restaurant_Name')


# In[ ]:


from wordcloud import WordCloud

plt.figure(figsize=(15, 4))
ip_string = ' '.join(data['Review'].dropna().to_list())

wc = WordCloud(background_color='white').generate(ip_string.lower())
plt.imshow(wc)


# In[ ]:




