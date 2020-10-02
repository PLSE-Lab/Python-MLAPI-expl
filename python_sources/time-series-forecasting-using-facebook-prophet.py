#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE

# 
# <table>
#   <tr><td>
#     <img src="https://drive.google.com/uc?id=1l7bHyrjzq839zVZE06cfdDksLabCN2hg"
#          alt="Fashion MNIST sprite"  width="1000">
#   </td></tr>
#   <tr><td align="center">
#     <b>Figure 1. Future Sales Time-series Prediction 
#   </td></tr>
# </table>
# 

# ![alt text](https://drive.google.com/uc?id=1vi45x-LGEzwvJoQstierOC1QZ11QQUmS)

# ![alt text](https://drive.google.com/uc?id=1eLLebiXwkN6x1dpsopQmkVNkR9zAYL7H)

# ![alt text](https://drive.google.com/uc?id=1a_q_DC8NyGBmcrxE0sGV4r6Hl-0w6G0K)

# ![alt text](https://drive.google.com/uc?id=1hNE0Wwc_bCCIO-AUAi6Xqo_9Bf1Xbh2o)

# ![alt text](https://drive.google.com/uc?id=1lQVgHsXn4Ur61dgYul1G-CmseLLUCEOB)

# # TASK #2: IMPORT LIBRARIES AND DATASET 

# In[ ]:


get_ipython().system('pip install fbprophet')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet


# TASK #2.1: IMPORT SALES TRAINING DATA

# In[ ]:


# You have to include the full link to the csv file containing your dataset
df_train = pd.read_csv('../input/rossmann-store-sales/train.csv')
df_train.head()


# #### almost a million observation 
# #### 1115 unique stores 
# #### Note that sales is the target variable (that's what we are trying to predict) 
# 
# #### Id: transaction ID (combination of Store and date) 
# #### Store: unique store Id
# #### Sales: sales/day, this is the target variable 
# #### Customers: number of customers on a given day
# #### Open: Boolean to say whether a store is open or closed (0 = closed, 1 = open)
# #### Promo: describes if store is running a promo on that day or not
# #### StateHoliday: indicate which state holiday (a = public holiday, b = Easter holiday, c = Christmas, 0 = None)
# #### SchoolHoliday: indicates if the (Store, Date) was affected by the closure of public schools
# #### Data Source: https://www.kaggle.com/c/rossmann-store-sales/data
# 
# 

# In[ ]:


df_train.info()
df_train.describe()


# #### 9 columns in total 
# #### 8 features, each contains 1017209 data points
# #### 1 target variable (sales)
# #### Average sales amount per day = 5773 Euros, minimum sales per day = 0, maximum sales per day = 41551 
# #### Average number of customers = 633, minimum number of customers = 0, maximum number of customers = 7388###

# TASK #2.2: IMPORT STORE INFORMATION DATA

# In[ ]:


df_store = pd.read_csv('../input/rossmann-store-sales/store.csv')
df_store.head()
# StoreType: categorical variable to indicate type of store (a, b, c, d)
# Assortment: describes an assortment level: a = basic, b = extra, c = extended
# CompetitionDistance (meters): distance to closest competitor store
# CompetitionOpenSince [Month/Year]: provides an estimate of the date when competition was open
# Promo2: Promo2 is a continuing and consecutive promotion for some stores (0 = store is not participating, 1 = store is participating)
# Promo2Since [Year/Week]: date when the store started participating in Promo2
# PromoInterval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store


# In[ ]:


df_store.info()
df_store.describe()


# #### Let's do the same for the store_info_df data
# #### Note that the previous dataframe includes the transactions recorded per day (in millions)
# #### This dataframe only includes information about the unique 1115 stores that are part of this study 
# #### on average, the competition distance is 5404 meters away (5.4 kms)

# # TASK #3: EXPLORE DATASET

# TASK #3.1: EXPLORE SALES TRAINING DATA

# In[ ]:


# Let's see if we have any missing data, luckily we don't!
sns.heatmap(df_train.isnull(),yticklabels=False,cmap='Blues',cbar=False)


# In[ ]:


df_train.hist(bins=30,color='g',figsize = (20,20))


# #### Average 600 customers per day, maximum is 4500 (note that we can't see the outlier at 7388!)
# #### Data is equally distibuted across various Days of the week (~150000 observations x 7 day = ~1.1 million observation) 
# #### Stores are open ~80% of the time
# #### Data is equally distributed among all stores (no bias)
# #### Promo #1 was running ~40% of the time 
# #### Average sales around 5000-6000 Euros
# #### School holidays are around ~18% of the time###

# In[ ]:


# Let's see how many stores are open and closed! 
df_train['Open'].value_counts()


# In[ ]:


# Count the number of stores that are open and closed
print('Total number of stores: {}'.format(len(df_train)))
print('Total number of open stores: 844392')
print('Total number of closed stores: 172817')


# In[ ]:


# only keep open stores and remove closed stores
df_train = df_train[df_train['Open']==1]


# In[ ]:


# Let's drop the open column since it has no meaning now
df_train.drop('Open',axis=1,inplace=True)


# In[ ]:


df_train.shape


# In[ ]:


df_train.describe()


# In[ ]:


# Average sales = 6955 Euros,	average number of customers = 762	(went up)


# TASK #3.2: EXPLORE STORES INFORMATION DATA

# In[ ]:


# Let's see if we have any missing data in the store information dataframe!
sns.heatmap(df_store.isnull(),cbar=False,cmap='Blues',yticklabels=False)


# In[ ]:


# Let's take a look at the missing values in the 'CompetitionDistance'
# Only 3 rows are missing 
df_store['CompetitionDistance'].isnull().sum()


# In[ ]:


df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].mean(),inplace=True)


# In[ ]:


# Let's take a look at the missing values in the 'CompetitionOpenSinceMonth'
# many rows are missing = 354 (almost one third of the 1115 stores)
df_store['CompetitionOpenSinceMonth'].isnull().sum()


# In[ ]:


# It seems like if 'promo2' is zero, 'promo2SinceWeek', 'Promo2SinceYear', and 'PromoInterval' information is set to zero
# There are 354 rows where 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth' is missing
# Let's set these values to zeros 
df_store.fillna(0,inplace=True)


# In[ ]:


sns.heatmap(df_store.isnull(),cbar=False,cmap='Blues',yticklabels=False)


# In[ ]:


# half of stores are involved in promo 2
# half of the stores have their competition at a distance of 0-3000m (3 kms away)
df_store.hist(figsize=(20,20),color='g',bins=30)


# TASK #3.3: EXPLORE MERGED DATASET 

# In[ ]:


# Let's merge both data frames together based on 'store'
merged_df = pd.merge(df_train,df_store,how='inner',on='Store')


# In[ ]:


merged_df.sample(5)


# In[ ]:


correlations = merged_df.corr()['Sales'].sort_values(ascending=False)
correlations
# customers and promo are positively correlated with the sales 
# Promo2 does not seem to be effective at all 


# In[ ]:


correlations = merged_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlations,annot=True,cmap="YlGnBu",linewidths=.5)
# Customers/Prmo2 and sales are strongly correlated 


# In[ ]:


# Let's separate the year and put it into a separate column 
merged_df['Year'] = pd.DatetimeIndex(merged_df['Date']).year
merged_df['Month'] = pd.DatetimeIndex(merged_df['Date']).month
merged_df['Day'] = pd.DatetimeIndex(merged_df['Date']).day


# In[ ]:


merged_df.head()


# In[ ]:


# Let's take a look at the average sales and number of customers per month 
# 'groupby' works great by grouping all the data that share the same month column, then obtain the mean of the sales column  
# It looks like sales and number of customers peak around christmas timeframe
axis = merged_df.groupby('Month')[['Sales']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.figure()
axis = merged_df.groupby('Month')[['Customers']].mean().plot(figsize=(10,5),marker='o',color='g')


# In[ ]:


# Let's take a look at the sales and customers per day of the month instead
# Minimum number of customers are generally around the 24th of the month 
# Most customers and sales are around 30th and 1st of the month
axis = merged_df.groupby('Day')[['Sales']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.figure()
axis = merged_df.groupby('Day')[['Customers']].mean().plot(figsize=(10,5),marker='o',color='g')


# In[ ]:


# Let's do the same for the day of the week  (note that 7 = Sunday)
axis = merged_df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.figure()
axis = merged_df.groupby('DayOfWeek')[['Customers']].mean().plot(figsize=(10,5),marker='o',color='g')


# In[ ]:


fig,ax = plt.subplots(figsize=(20,10))
merged_df.groupby(['Date','StoreType']).mean()['Sales'].unstack().plot(ax=ax)


# #### As seen that the store type b has the best sales and type a has the least

# In[ ]:


plt.figure(figsize=(15,10))

plt.subplot(211)
sns.barplot(x="Promo",y='Sales',data=merged_df)
plt.subplot(212)
sns.barplot(x="Promo",y='Customers',data=merged_df)


# In[ ]:


plt.figure(figsize=(15,10))

plt.subplot(211)
sns.violinplot(x="Promo",y='Sales',data=merged_df)
plt.subplot(212)
sns.violinplot(x="Promo",y='Customers',data=merged_df)


# # TASK #4: UNDERSTAND THE INTUITION BEHIND FACEBOOK PROPHET

# ![alt text](https://drive.google.com/uc?id=1I4lBgLaqERF_-lpGYLuht02wJmwcLGG-)

# ![alt text](https://drive.google.com/uc?id=1CZ24f-TbnRzaXV9Arke0fNTUm7Kon1gK)

# ![alt text](https://drive.google.com/uc?id=16gaoTeeuU5PxNZRHt8n2XyFJ52ft1xb7)

# # TASK #5: TRAIN THE MODEL PART A

# In[ ]:


def sales_prediction(Store_ID, sales_df, periods):
  # Function that takes in the data frame, storeID, and number of future period forecast
  # The function then generates date/sales columns in Prophet format
  # The function then makes time series predictions

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')
  
  model    = Prophet()
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods=periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)


# In[ ]:


sales_prediction(10, merged_df, 60)


# **Now I will consider holidays too in order to make predictions more real**

# # TASK #6: TRAIN THE MODEL PART B

# 
#    - StateHoliday: indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
#    - SchoolHoliday: indicates if the (Store, Date) was affected by the closure of public schools
#   
# 
# 
# 
# 
# 

# In[ ]:


def sales_prediction_better(Store_ID, sales_df, holidays, periods):
  # Function that takes in the storeID and returns two date/sales columns in Prophet format
  # Format data to fit prophet 

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')
  
  model    = Prophet(holidays = holidays)
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods = periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)


# In[ ]:


# Get all the dates pertaining to school holidays 
school_holidays = merged_df[merged_df['SchoolHoliday'] == 1].loc[:, 'Date'].values
school_holidays.shape


# In[ ]:


# Get all the dates pertaining to state holidays 
state_holidays = merged_df[ (merged_df['StateHoliday'] == 'a') | (merged_df['StateHoliday'] == 'b') | (merged_df['StateHoliday'] == 'c')  ].loc[:, 'Date'].values
state_holidays.shape


# In[ ]:


state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                               'holiday': 'state_holiday'})
school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                                'holiday': 'school_holiday'})


# In[ ]:


# concatenate both school and state holidays 
school_state_holidays = pd.concat((state_holidays, school_holidays))


# In[ ]:


# Let's make predictions using holidays for a specific store
sales_prediction_better(14, merged_df, school_state_holidays, 60)


# **The results of predictions are good. Using Facebook prophet is simple, that what I like about it. But I wouold still like to solve this problem statement using ARIMA & LSTM. Then would like to compare results.**

# In[ ]:




