#!/usr/bin/env python
# coding: utf-8

# # ***** Exploratory Analysis of Indian Startups Funding *****

# <img src='https://drive.google.com/uc?id=1jWwsVdwxE0JIDRvXsjTvNDT4VE-2r-Cc' widht=500 >

# ### Introduction= This notebook is analysis some important questions which are related to startups India:-
# * Q1). Number of unique Startups.
# * Q2). Top 10 Startups which are attrative means they have maximum number of Investors.
# * Q3). Top 10 Startups with maximum funding.
# * Q4). Top 10 Startups with minimum funding.
# * Q5). Number of Unique Investors.
# * Q6). Top 10 Investors in Indian ecosystem according to the total amount they invested in startups.
# * Q7). Top 10 industries which are favourite of Investors.
# * Q8). Top 10 Sub-industries in Technology industry according to number of times Investors invested in Startups.
# * Q9). Top 10 Industry where funding is maximum.
# * Q10). Top 10 cities which have maximum startups.
# * Q11). Top 10 cities which got maximum funding and their funding distribution.
# * Q12). Funding distribution in Indian Startups over Years.
# * Q13). Number of startups over months and over years.
# * Q14). Different type of funding (Investment type).

# In[24]:


# Importing useful Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


# Importing dataset

try:
    df_startup = pd.read_csv('input/startup_funding.csv')
    
except Exception as e:
    df_startup = pd.read_csv('../input/startup_funding.csv')


# In[26]:


# Let's check top 5 entries.

df_startup.head()


# In[27]:


# Basic information.

print(df_startup.info())


# In[28]:


# Dealing with missing data.

missing_data = df_startup.isnull().sum().sort_values(ascending=False)
missing_data = (missing_data/len(df_startup))*100
print(missing_data)


# * 82% of Remarks are full of NaN, hence we can remove this column. It will not effect our results.

# In[29]:


# Removing unwanted column.

try:
    df_startup.drop('Remarks', axis=1, inplace=True)
except Exception:
    pass


# In[30]:


# Converting AmountInUSD column into integer type as it in string type.
print(f"The Type of AmountInUSD column is\t {type(df_startup['AmountInUSD'][0])}")
print("We have to convert it into integer type.")

df_startup['AmountInUSD'] = df_startup['AmountInUSD'].apply(lambda x: float(str(x).replace(",","")))
df_startup['AmountInUSD'] = pd.to_numeric(df_startup['AmountInUSD'])

print(f"The type of AmountInUSD column after conversion is\t {type(df_startup['AmountInUSD'][0])}")


# In[31]:


# Checking dataset after cleaning it.

df_startup.head()


# ### ======================================================================================

# # STARTUPS ANALYSIS

# ### Q1). Number of unique Startups.

# In[32]:


print(f"The numebr of Unique Startups are\t {df_startup['StartupName'].nunique()} ")


# ### Q2). Top 10 Startups which are attrative means they have maximum number of Investors.

# In[33]:



df_startup['StartupName'].value_counts()[:10].plot(kind='bar', figsize=(10,5))
plt.show()


# * Swiggy. Ubercap, Jugnoo they are in high demand by Investor.
# * Maximum number of times startups got one time investment.

# #### ===================================================================================================

# ### Q3). Top 10 Startups with maximum funding.

# In[34]:


temp_df = df_startup.sort_values('AmountInUSD', ascending=False, )
temp_df = temp_df[['StartupName', 'AmountInUSD']][:10].set_index('StartupName', drop=True, )
temp_df.plot(kind='bar', figsize=(10,5))
plt.show()

temp_df.plot(kind='pie', subplots=True, figsize=(12,6), autopct='%.f%%')
plt.show()

temp_df.T


# #### ====================================================================================================

# ### Q4). Top 10 Startups with minimum funding.

# In[35]:


new_df = df_startup[df_startup['AmountInUSD'].notnull()]
new_df.sort_values('AmountInUSD', inplace=True)
new_df = new_df[['StartupName', 'AmountInUSD']][:10].set_index('StartupName', drop=True)
new_df.plot(kind='bar', figsize=(10,5))
plt.show()
new_df.T


# #### ===================================================================================================

# # INVESTORS AND INDUSTRIES ANALYSIS

# ### Q5). Number of Unique Investors.

# In[36]:


print(f"The number of unique investors in Indian ecosystem between 2015 to 2017 are\t {df_startup['InvestorsName'].nunique()}")


# <img src='https://drive.google.com/uc?id=1xWtWww9hQoVBaGVFzAe7WhpFUrOJLjv9' width=500>

# ### Q6). Top 10 Investors in Indian ecosystem according to the total amount they invested in startups.

# In[37]:


investor = df_startup.groupby('InvestorsName')['AmountInUSD'].sum().reset_index()
investor.sort_values('AmountInUSD', inplace=True, ascending=False)
investor.reset_index()[:10]


# ### ======================================================================================

# ### Q7). Top 10 industries which are favourite of Investors.

# In[38]:


df_startup['IndustryVertical'].value_counts()[:10].reset_index()


# * Consumer internet and Technology industries are in high demand by Investors.
# * Their future is good.
# * If one want to start a new startup then they should start in these industries.
# * It's good to see education industry in top 10

# #### ====================================================================================================

# ### Q8). Top 10 Sub-industries in Technology industry according to number of times Investors invested in Startups.

# In[39]:


temp_df = df_startup[df_startup['IndustryVertical'].isin(['Technology','technology'])]
temp_df = temp_df['SubVertical'].value_counts()[:10].reset_index()
temp_df.columns = ['Sub Industry', 'Number of times']
temp_df.set_index('Sub Industry', drop=True, inplace=True)
temp_df.plot(kind='bar', figsize=(10,5))
plt.show()
temp_df.T


# <img src='https://drive.google.com/uc?id=1Mxy8kFijaRZJlcWtUIxxFI6t1-ZfJjkT' width=500>

# #### ====================================================================================================

# ### Q9). Top 10 Industry where funding is maximum.

# In[40]:


# converting ecommerce into Ecommerce
df_startup['IndustryVertical'] = df_startup['IndustryVertical'].apply(lambda x: 'ECommerce' if x=='eCommerce' else x)

new_df = df_startup.groupby('IndustryVertical')['AmountInUSD'].sum().reset_index()
new_df.sort_values('AmountInUSD', inplace=True, ascending=False)

new_df.set_index('IndustryVertical', inplace=True, drop=True)
new_df[:10].plot(kind='bar', figsize=(10,5))
plt.show()

new_df[:10].plot(kind='pie', subplots=True, figsize=(12,6), autopct='%.f%%')
plt.show()

new_df[:10].T


# * ECommerce got maximum amount of funding followed by Consumer Internet and then by Technology.
# * If one want to start a startup then they can start in these industries.

# #### ==================================================================================================

# # CITIES ANALYSIS

# ### Q10). Top 10 cities which have maximum startups.

# In[41]:


city = df_startup['CityLocation'].value_counts()[:10].reset_index()
city.columns = ['City', 'Number of Startups']
city.set_index('City', drop=True, inplace=True)
city.plot(kind='bar', figsize=(10,5), title='Top 10 Cities which have maximum startups')
plt.show()

city.plot(kind='pie', subplots=True, figsize=(12,6), autopct='%.f%%')
plt.show()
city.T


# * Bangalore,the silicon valley of India, is on top followed by Mumbai and then by Delhi.
# * Jaipur is also in top 10 with 25 startups. A good start from Jaipur. 
# * Jaipur can be a hub of startups as there is huge space avilable in Jaipur and government is also come out with new schemes.

# #### ====================================================================================================

# ### Q11). Top 10 cities which got maximum funding and their funding distribution.

# In[42]:


# Dealing with duplicates name.
l = ['Bangalore', 'Bangalore/ Bangkok', 'SFO / Bangalore','Seattle / Bangalore', 'Bangalore / SFO',
     'Bangalore / Palo Alto', 'Bangalore / San Mateo', 'Bangalore / USA',   ]
df_startup['CityLocation'] = df_startup['CityLocation'].apply(lambda x: 'Bangalore' if x in l else x )

city_df = df_startup.groupby('CityLocation')['AmountInUSD'].sum().reset_index()
city_df.sort_values('AmountInUSD', ascending=False, inplace=True)
city_df = city_df[:10]
city_df.reset_index(inplace=True, drop=True)
city_df.set_index('CityLocation', inplace=True, drop=True)
city_df.plot(kind='bar', figsize=(12,6), title='Top 10 cities which got maximum funding')
plt.show()

city_df.plot(kind='pie', figsize=(12,6), autopct='%.f%%', subplots=True)
plt.show()

city_df.T


# * 50% of funding was done in Bangalore city.
# * Mumbai have so many startups but have less funding.
# 

# #### Let's check out their distribution.

# In[43]:


# top_city is the cities which have maximum funding.
top_city = city_df.index
temp_df = df_startup[df_startup['CityLocation'].isin(top_city)]
temp_df = temp_df[['CityLocation', 'AmountInUSD']]

plt.figure(figsize=(12,6))
sns.swarmplot(data=temp_df, x=temp_df['CityLocation'], y=temp_df['AmountInUSD'])

plt.show()


# * As one can see there are so many outliers in Bangalore and in New Delhi.

# #### =====================================================================================================

# # FUND ANALYSIS

# ### Q12). Funding distribution in Indian Startups over Years.

# In[44]:


# Converting some Dates into its proper format, as they were entered wrong.
def convert(x):
    if x=='12/05.2015':
        return '12/05/2015'
    elif x=='13/04.2015':
        return '13/04/2015'
    elif x=='15/01.2015':
        return '15/01/2015'
    elif x=='22/01//2015':
        return '22/01/2015'
    else:
        return x

df_startup['Date'] = df_startup['Date'].apply(convert)

# Need to convert string into datetime format object.
df_startup['year'] = (pd.to_datetime(df_startup['Date']).dt.year)
df_startup.head()

plt.figure(figsize=(12,6))
sns.boxenplot(data=df_startup, x='year', y='AmountInUSD')
plt.show()


# * There is no outliers in 2016 as compare to 2015 and 2017.
# * In 2017 we are getting left skewness , means many large amounts were invested in 2017.

# #### ====================================================================================================

# ### Q13). Number of startups over months and over years.

# In[45]:


# New column with date in year-month format.

df_startup['year_month'] = (pd.to_datetime(df_startup['Date']).dt.year*100) + (pd.to_datetime(df_startup['Date']).dt.month)

times = df_startup['year_month'].value_counts().reset_index()
times.set_index('index', drop=True, inplace=True)
times.index.name = 'Month-Year'
times.columns = ['Number of Startups']
times.sort_index(inplace=True)
times.plot(kind='bar', figsize=(13,7), title='Number of Startups over month')
plt.show()

# Let's see number of startups over Years
df_startup['year'].value_counts().plot(kind='bar', figsize=(13,7), title='Number of Startups over Years')
plt.show()
df_startup['year'].value_counts().plot(kind='pie', figsize=(11,7), title='Number of Startups over Years', subplots=True, autopct='%.f%%')
plt.show()


# * The trend in startups first incres in 2016 and then decrease in 2017.
# 

# #### ====================================================================================================

# ### Q14). Different type of funding (Investment type).

# In[46]:


# Dealing with incorrect entries.
def convert(x):
    if x== 'SeedFunding':
        return 'Seed Funding'
    elif x== 'PrivateEquity':
        return 'Private Equity'
    elif x== 'Crowd funding':
        return 'Crowd Funding'
    else:
        return x

df_startup['InvestmentType'] = df_startup['InvestmentType'].apply(convert)

df_startup['InvestmentType'].value_counts().plot(kind='bar', figsize=(12,6), title='Type of Investment')
plt.show()

df_startup['InvestmentType'].value_counts().plot(kind='pie', figsize=(12,6), subplots=True, autopct='%.f%%')
plt.show()


# #### ===================================================================================================
# #### ===================================================================================================
# 

# # ->  If this kernel is Useful, then Please Upvote
# <img src='https://drive.google.com/uc?id=1Q8awTKg6LtoK5gIPUVc4c2SZ6Kz0g5-h'>

# In[ ]:




