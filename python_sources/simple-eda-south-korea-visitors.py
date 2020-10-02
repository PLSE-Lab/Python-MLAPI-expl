#!/usr/bin/env python
# coding: utf-8

# # South Korea Visitors
# ## Foreign visitors into South Korea
# 
# ![](https://images.unsplash.com/photo-1548115184-bc6544d06a58?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80)
# 
# This dataset deals with the visitors of foreigners into South Korea.  
# It includes foreigners(not Korean), overseas Koreans and crew members, except for some of the foreign arrivals who are not considered tourists (diplomats, soldiers, permanent residents, visiting cohabitation and residence).  
# 
# 
# It covers data from January 2019 to April 2020 and it shows the number of visitors from 60 countries.  
# In addition, the number of visitors is classified according to gender/age/purpos

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
data = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data.append(os.path.join(dirname, filename))


# In[ ]:


data


# # Data

# ## age

# In[ ]:


# age
df = pd.read_csv(data[0])
df


# In[ ]:


df.shape


# ## gender

# In[ ]:


df = pd.read_csv(data[1])
df


# In[ ]:


df.shape


# ## purpose

# In[ ]:


df = pd.read_csv(data[2])
df


# In[ ]:


df.shape


# # Simple EDA

# In[ ]:


# using age data
df = pd.read_csv(data[2])


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.groupby('nation').date.count()


# There are 16 data in each country.
#   
#   16 months (2019 + 2020 (1-4))

# Find the average number of visitors from each country.

# In[ ]:


df.groupby('nation').mean()['visitor'].sort_values(ascending=False)


# The average number of visitors can be seen in the order of China, Japan, Taiwan, USA, Hong Kong and Thailand.  
# Given that countries other than the U.S. are almost in Asian countries, it is possible to predict that the number of visitors from nearby countries will account for a large portion.

# # Visitor

# Let's visualize the number of visitors in every country.

# In[ ]:


def all_graph(df, x, y, length):
    fig,axes = plt.subplots(1,1,figsize=(20, 16))
    axes.set_title(y)
    axes.set_ylabel(y)
    axes.set_xlabel(x)
    axes.set_xticklabels(df[x].unique(), rotation=45)
    qualitative_colors = sns.color_palette("Paired", length)
    sns.lineplot(x, y, ci=None, hue='nation', 
                 marker='o', data=df, linewidth=2, palette=qualitative_colors)
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:


all_graph(df, 'date', 'visitor', 60)


# The previous results showed that China, Japan, Taiwan and the United States accounted for most of the visitors to Korea.
#   
#   Then let's take a closer look at this major country.

# In[ ]:


top_countries = ['China', 'Japan', 'Taiwan', 'USA']


# In[ ]:


def time_visitor_graph(name):
    fig,axes = plt.subplots(1,1,figsize=(10, 8))
    x = df[df['nation']==name].date
    y = df[df['nation']==name].visitor
    axes.set_title(name)
    axes.set_ylabel("The number of visitors")
    axes.set_xlabel("Date")
    axes.set_xticklabels(x, rotation=45)
    axes.plot(x, y, linewidth=3.0)


# In[ ]:


for country in top_countries:
    time_visitor_graph(country)


# I have visualized the monthly number of visitors to Korea, including China, Japan, Taiwan and the United States.  
# 
# Although the number of visitors decreased and increased in different countries, it can be seen that the number of visitors in all countries has decreased significantly due to the Covid-19.

# Let's directly compare the January-April 2019 data with the January-April 2020 data.

# In[ ]:


def month_compare_graph(name):
    fig,axes = plt.subplots(1,1,figsize=(10, 8))
    x = [1, 2, 3, 4]
    y = df[(df['date'].str.endswith(('-1', '-2', '-3', '-4'))) & (df['nation'] == name)].visitor
    
    axes.set_title(name)
    axes.set_ylabel("The number of visitors")
    axes.set_xlabel("Month")
    axes.plot(x, y[:4], c='b', linewidth=5.0, label='2019')
    axes.plot(x, y[4:], c='r', linewidth=5.0, label='2020')
    axes.legend(loc=3)


# In[ ]:


for country in top_countries:
    month_compare_graph(country)


# # Growth

# Growth means growth percentage in the number of visitors compared to the same month last year

# In[ ]:


all_graph(df, 'date', 'growth', 60)


# Since there are 60 countries, it is not easy to see the graph.  
# Use only the 10 countries with the highest average number of visitors.

# In[ ]:


top_countries = df.groupby('nation').mean()['visitor'].sort_values(ascending=False)[:10].index
top_countries


# In[ ]:


df_top = df[df['nation'].isin(top_countries)]
df_top


# In[ ]:


all_graph(df_top, 'date', 'growth', 10)


# Let's look at the growth rate using `purpose` dataset.

# In[ ]:


df = pd.read_csv(data[1])


# In[ ]:


df.sort_values('growth', ascending=False).head(5)


# In[ ]:


df.sort_values('growth').head(5)


# Macau had the highest and lowest growth, and Hong Kong had the second highest and lowest growth.  
# 
# What's more surprising is that the growth was the highest in January, when the first confirmed case of covid-19 was made in Korea. (It was at the end of January when the confirmed person came out, so I think it wouldn't have affected much.)
# 
# 
# The year 2020 is covid-19, with the growth rate falling to negative compared to last year. Let's look at the data except 2020.

# In[ ]:


df_2019 = df[df['date'].str.startswith('2019')]


# In[ ]:


df_2019.sort_values('growth', ascending=False).head(5)


# In[ ]:


df_2019.sort_values('growth').head(5)


# In my opinion, when the data on overseas Koreans is compiled, the data is not properly aggregated by purpose, so it has all gone to 'others'.
# 
# 
# In addition, the decrease in the number of visitors from countries such as Norway, Iran and Switzerland between January and February of 19 seems to have decreased due to the influx of tourists during the PyeongChang 2018 Olympic Winter Games.

# # share
# 
# It also looks at data from 10 major countries.

# In[ ]:


all_graph(df_top, 'date', 'share', 10)


# Interesting. Visitors, which were previously focused on China and Japan, can be seen in recent years when covid-19 has become worse, with Malaysia and the United States accounting for a large portion.
# 
# 
# Japan accounts for a large portion in February 2020. We can see a sharp decline in April.
