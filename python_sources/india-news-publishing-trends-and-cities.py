#!/usr/bin/env python
# coding: utf-8

# This kernal provides an overview of the publishing volumes, topics and trends for **Times of India**, India's largest news website.
# 
# Times of India was founded in 1836 and started a daily edition in 1850. Today it is the highest selling English language daily in the world.

# In[ ]:


"""
@author: Rohit Kulkarni
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print (pd.__version__)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/india-news-headlines.csv", dtype={'publish_date': object})

df['publish_month'] = df.publish_date.str[:6]
df['publish_year'] = df.publish_date.str[:4]
df['publish_month_only'] = df.publish_date.str[4:6]
df['publish_day_only'] = df.publish_date.str[6:8]

df['dt_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')
df['dt_month'] = pd.to_datetime(df['publish_month'], format='%Y%m')

print (df.info())


# **Articles Published Per Month Over 19.5 Years**

# In[ ]:


#Monthly and daily plot

grp_date = df.groupby(['dt_date'])['headline_text'].count()
grp_month = df.groupby(['dt_month'])['headline_text'].count()

ts = pd.Series(grp_date)
ts.plot(kind='line', figsize=(20,10),title='Articles per day')
#plt.show()

ts = pd.Series(grp_month)
ts.plot(kind='line', figsize=(20,10),title='Articles per month')
plt.show()


# **Monthly Publishing Rate with Each Year Overlapping**

# In[ ]:


#Year slice plotting

years=df['publish_year'].unique().tolist()
print (years)

for year in years:
    yr_slice = df.loc[df.publish_year==year]
    grp_month = yr_slice.groupby(['publish_month_only'])['headline_text'].count()
    month_ts = pd.Series(grp_month)
    month_ts.plot(kind='line', figsize=(20,10), style='o-', legend=True, label=year)
    
plt.show()


# **Number of Articles Per Indian City**
# 
# This represents around 1.84 million articles or around 56% of the total records.

# In[ ]:


#City slice plotting

df_city = df[df['headline_category'].str.contains('^city\.[a-z]+$', regex=True)]
df_city['city_name'] = df_city.headline_category.str[5:]

city_list=df_city['city_name'].unique().tolist()
print (city_list)

#bar plot of all cities
grp_city = df_city.groupby(['city_name'])['headline_text'].count().nlargest(50)
ts = pd.Series(grp_city)
ts.plot(kind='bar', figsize=(20,10),title='Articles per city')
plt.show()


# **Share of Articles By Top 40 Covered Cities**

# In[ ]:


#pie chart of top 40 cities
grp_top_city = df_city.groupby(['city_name'])['headline_text'].count().nlargest(40)
ts = pd.Series(grp_top_city)
ts.plot(kind='pie', figsize=(10,10),title='Share of Top 40 Cities')
plt.show()


# **Articles with Categories not related to Cities**
# 
# This represents the remaining 1.45 million articles that fall into categories like international news, bollywood etc.
# 
# Around 200k are of unknown category and another 5K were removed by me for being too verbose. Can you classify these using ML?

# In[ ]:


# Non city related categories

df_non_city = df[~df['headline_category'].str.contains('city', regex=False)]

grp_non_city = df_non_city.groupby(['headline_category'])['headline_text'].count().nlargest(25)
ts = pd.Series(grp_non_city)
ts.plot(kind='bar', figsize=(20,10),title='Top 25 Non-city Categories')
plt.show()

