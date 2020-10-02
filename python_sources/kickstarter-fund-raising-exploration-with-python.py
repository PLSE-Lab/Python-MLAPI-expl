#!/usr/bin/env python
# coding: utf-8

# The datasets contain kickstarter project from 05/2009 to 01/2018.
# 
# Crowdfunding has been so popular in the startup world, it helped a lot of people to realize their dreams. And Kickstarter is no doubt one of the best-known. So I want to dig deep in its data and find out how it had helped people in the past.
# 
# Before I started this project. Based on my knowledge and past experience. Kickstarter has always been one of the top crowdsourcing website for startups. I assumed that it is a site for high-tech startups, since a lot of its top-funded legends (Pebble,BAUBAX,OUYA etc) are all tech related. Those projects have raised millions of dollars.
# 
# However,from a data science perspective, I know that I need to analyze the site in a more logical and objective way. This exercise is about doing some simple exploration and look into the truth of fund raising in Kickstarter. 

# In[ ]:


import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


df = pd.read_csv('../input/ks-projects-201801.csv')


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


df['main_category'].describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


state_color = ["#FD2E2E", "#E6E6E6", "#17B978", "#CFCBF1", "#4D4545", "#588D9C"]
sns.factorplot('state',data=df,kind='count', size=10,palette=state_color)


# In[ ]:


group_state = df.groupby(['state'])


# In[ ]:


df['state'].value_counts()


# In[ ]:


Success_Rate = 133956/378661
Success_Rate


# **Only little over of 35% all projects were successful.**

# In[ ]:


sns.factorplot('main_category',data=df,kind='count',hue='state' ,size=15, palette=state_color)


# From the chart above, we could see that only Music,Comics, Theater and Dance have more successful projects than failed projects. The crowdfunding legends from technology were just a tiny part of Kickstart projects. Kickstart has supported art related fields in a significant way. And crowdfunding on kickstart might not be a good option for tech companies due to the low success rate. It is also noticed that a great amount of projects are for Film and Videos. 

# In[ ]:


group_maincategories = df.groupby(['main_category'])
group_maincategories['main_category'].count().reset_index(name='count').sort_values(['count'], ascending=False)


# Surprisely, technology only ranked 5th for the total number of projects on Kickstarter. A lot of peole come for Film&Video, Music, Publishing and Games.
# 
# But why technology projects are so easy to fail while some projects tend to succeed. Maybe the goals are too high for tech related projects, since it require a lot more investment. And I am going to verify that.

# In[ ]:


goal_fund = df['goal'].groupby(df['main_category'])
goal_fund.mean().reset_index(name='mean').sort_values(['mean'], ascending=False)


# Clearly from this chart, I noticed that tech related projects have an average goal of over 110,000 dollars, which is more than ten times as Dance.

# In[ ]:


df['usd pledged'].describe()
#lets take out the null value so the average for usd pledged would be more precise 


# The average usd pledged is only a little over **7000**, and more than **75%** of the projects pledged less than **3000** dollars.
# 
# WOW for less than I expected!

# In[ ]:


df_tech = df[df['main_category']=='Technology']
df_tech.head()


# In[ ]:


df_tech['goal'].describe()


# In[ ]:


df_tech['usd pledged'].quantile(0.90)


# In[ ]:


df_tech[df_tech['usd pledged']>119712].count()


# Only **0.7%** project has reached the avergae goal of 119,712 USD. 

#  **Conclusion**
#  
#  It is super hard for tech companies to succeed on Kickstarter. And even you managed to raise fund on Kickstarter, it is unlikely that you could raise sufficient money for your project. If you are a tech company and is turning to Kickstarter for your startup, you'd better be prepared. If you are doing art projects and need relative less money, Kickstarter might be a good help. 

# In[ ]:




