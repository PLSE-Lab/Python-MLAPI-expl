#!/usr/bin/env python
# coding: utf-8

# # About Kiva :
# 
# ![Kiva](https://designtoimprovelife.dk/wp-content/uploads/2009/08/KIVA.jpg)
# 
# <font size=5>Basically,</font> Kiva is a nonprofit organization which offer a platform that you can simply lend 25$ to them and they'll help those in need to go to school, start a business, or get the basic elements they need for their lives with 0% interest loan. The most important thing is that Kiva ensure their borrowers are on the right track. Kiva keep track of those borrowers' progress. For me, this cannot be better. I'm now serving my military duty. I get along with those who don't have further education than junior high school or even elementary school. I found some of them are eager for a better life. I realize that I'm lucky that I have enough resource to help myself getting a better life by myself. However, they don't. They may be born in ghetto or surrounded by gangs, like Ezekiel in [the get down](https://en.wikipedia.org/wiki/The_Get_Down). Kiva can be the Papa Fuerte for them, giving them an opportunity to earn themselves a better life.
# 
# Therefore, I want to explore this data and know more about the situation, even help them. And this is the reason why I want to be a data scientist, to maximize my positive impact on this world as more as possible.

# ## Before you start reading, you might want to see [My another work about Kiva Crowdfunding](https://www.kaggle.com/justjun0321/help-making-this-world-better-borrowers-analysis)

# ## Outline :
# 
# * **Loading and Assessing data**
# 
# * **Columns overlook**
# 
# * **Time Series Exploding**
# 
# * **Arima**

# ## Loading and Assessing data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from statsmodels.graphics import tsaplots
import statsmodels.api as sm


# In[2]:


df = pd.read_csv('../input/kiva_loans.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# ** As I assumed, the datatype of those time columns are objects instead of datetime. So I need to fix this **

# ** Before I start, let me check the NA values in this dataset **

# In[5]:


pd.isnull(df).sum()


# ** There are a lot of NA values in this dataset. For me, since I'm going to analyze the time series, I need to fix the NA values in those columns about time first. **

# In[6]:


df.funded_time = df['funded_time'].fillna(df['posted_time'])


# ** Since most of the funded_time is few hours after the posted_time, I'll straightly fill them with posted time **

# In[7]:


pd.isnull(df).sum()


# In[8]:


df.posted_time = np.array(df.posted_time,dtype='datetime64[D]')
df.disbursed_time = np.array(df.disbursed_time,dtype='datetime64[D]')
df.funded_time = np.array(df.funded_time,dtype='datetime64[D]')
df.date = np.array(df.date,dtype='datetime64[D]')


# ** Let me check if this works **

# In[9]:


df.info()


# ** Nice! We are ready! **

# # Columns overlook

# In[10]:


df.head()


# ** Here I see some columns I'm interested **
# 
# * ** funded_amount ** - funded_amount is one of the most important column in this dataset. I'll explore things around it a lot.
# 
# * **country** - I might subset data into different countries to look deeper.
# 
# * **lender_count** - I'll analyze around lender_count as funded_amount since I think the changing of the amount of lender may be interesting.
# 
# * **All the timestamp columns** - These are the keys of this kernels.

# # Time Series Exploding

# ## Disribution 

# ** First, I subset the date and funded amount, which are two main columns I want to explore for now. **

# In[11]:


df_sub = df[['funded_amount','date']]
df_sub = df_sub.set_index('date')


# In[12]:


plt.style.use('fivethirtyeight')

ax = df_sub.plot(linewidth=1,figsize=(20, 6), fontsize=10)
ax.set_xlabel('Date')
plt.show()


# ** Now, let me dig deeper during 2014 **

# In[13]:


df_sub_1 = df_sub['2014-01-01':'2014-12-31']

ax = df_sub_1.plot(linewidth=1,figsize=(20, 6), fontsize=10)
ax.set_xlabel('Date')

ax.axvline('2014-06-10', color='red', linestyle='-',linewidth=3,alpha = 0.3)

plt.show()


# ** There are few peaks in 2014, especially the one  that happened at 06-10 **

# In[14]:


mov_avg = df_sub.rolling(window=52).mean()

mstd = df_sub.rolling(window=52).std()

mov_avg['upper'] = mov_avg['funded_amount'] + (2 * mstd['funded_amount'])
mov_avg['lower'] = mov_avg['funded_amount'] - (2 * mstd['funded_amount'])

ax = mov_avg.plot(linewidth=0.8,figsize=(20, 6) , fontsize=10,alpha = 0.5)
ax.set_title('Rolling mean and variance of Fund \n from 2013 to 2017', fontsize=10)


# ** There are LOTS of noise in this plot. Therefore, I now aggregate it into weeks and plot it in 4 weeks as moving average ** 

# In[15]:


index = df_sub.index.week

df_sub_week = df_sub.groupby(index).mean()

mov_avg = df_sub_week.rolling(window=4).mean()

mstd = df_sub_week.rolling(window=4).std()

mov_avg['upper'] = mov_avg['funded_amount'] + (2 * mstd['funded_amount'])
mov_avg['lower'] = mov_avg['funded_amount'] - (2 * mstd['funded_amount'])

ax = mov_avg.plot(linewidth=0.8,figsize=(20, 6) , fontsize=10)
ax.set_title('Rolling mean and variance of Fund \n from 2013 to 2017 in weeks', fontsize=10)


# ** Much better for me :) **

# ### I want to explore more about the distribution of funds

# In[16]:


ax = plt.subplot()

ax.boxplot(df_sub['funded_amount'])
ax.set_yscale('log')
ax.set_xlabel('fund')
ax.set_title('Distribution of funds', fontsize=10)
plt.show()


# In[17]:


df_sub.describe()


# In[18]:


ax = df_sub.plot(kind='density',figsize=(20, 6) , linewidth=3, fontsize=10)
ax.set_xlabel('fund')
ax.set_ylabel('density')
ax.set_xscale('log')
ax.set_title('Density of funds', fontsize=10)
plt.show()


# ** To know more about the distribution of fund amount, I use boxplot with a statistical table and a density plot to know more. **
# 
# ** I can see that 75% of the funds are less than 900 but still some are between 100,000 and 1,000.**
# 
# ** In my opinion, after seeing the density plot, I think Kiva do plan on how much loans they want to issue with funds less than 1,000 but not for those over 1,000. **

# ## Autocorelation plot

# In[19]:


fig = tsaplots.plot_acf(df_sub_week['funded_amount'], lags=24)
plt.show()


# ** I can see that the size of funds are affected by those recent values **
# 
# ** And let's start it to see more about trends and seasonality **

# In[20]:


df_sub_mean = df_sub.groupby(pd.TimeGrouper('D')).mean().dropna()
df_sub_total = df_sub.groupby(pd.TimeGrouper('D')).sum().dropna()


# In[21]:


decomposition = sm.tsa.seasonal_decompose(df_sub_total)

trend = decomposition.trend

ax = trend.plot(figsize=(20, 6), fontsize=6)

ax.set_xlabel('Date', fontsize=10)
ax.set_title('Seasonal component of total fund', fontsize=10)
plt.show()


# ** As I can see, there are peaks and fluctuations from time to time **
# 
# ** Since I need to look about other columns, now I'll transform the datatype of the original dataframe **

# In[22]:


df['date'] = pd.to_datetime(df['date'])

df = df.set_index('date')


# In[23]:


new_df = df[['funded_amount','lender_count']]


# ### Mean fund for each day

# In[24]:


new_df_mean = new_df.groupby(pd.TimeGrouper('D')).mean().dropna()
new_df_total = new_df.groupby(pd.TimeGrouper('D')).sum().dropna()

new_df_mean.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()


# ### What I found is that there is a significant difference in around 2015-11. Beside that, most of the lender amount is strongly correlated to fund amount

# ### Total fund for each day

# In[25]:


new_df_total.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()


# ### However, I don't see that in total amount plot

# ### Mean fund for each week

# In[26]:


new_df_mean = new_df.groupby(pd.TimeGrouper('W')).mean().dropna()
new_df_total = new_df.groupby(pd.TimeGrouper('W')).sum().dropna()

new_df_mean.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()


# ### Total fund for each week

# In[27]:


new_df_total.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()


# ### Once I aggregate the data by week, the seasonality is much obvious.

# ### Mean fund for each month

# In[28]:


new_df_mean = new_df.groupby(pd.TimeGrouper('M')).mean().dropna()
new_df_total = new_df.groupby(pd.TimeGrouper('M')).sum().dropna()

new_df_mean.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()


# ### Total fund for each month

# In[29]:


new_df_total.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()


# ### But once I aggregate by month, I lost most of the seasonality

# ## Here I want to know more about the time spent on the application of loans and analyze it

# In[31]:


df['Time_Spent'] = df['posted_time'] - df['disbursed_time']


# In[34]:


df.head()


# ## It's sweet that Python made the "Time_Spent" column in this form. However, I need to check the datatype of this

# In[35]:


type(df.Time_Spent)


# In[ ]:





# # Unfortunately, I can only finish the first three part of this kernel since I'm serving my military duty and I'm called back by my base now. I won't be able to update it until next day off. If you have any comment or suggestion, please let me know. I'll appreciate for upvotes ;)

# In[ ]:




