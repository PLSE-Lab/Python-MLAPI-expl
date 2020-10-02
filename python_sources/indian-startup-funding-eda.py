#!/usr/bin/env python
# coding: utf-8

# ## Hello Kagglers!!
# 
# ### Today we will explore use Indian startup funding data to explore and gain knowledge about funding investment market. This will help us see wheather is India had heathly market for startup culture and their growth  
# 
# Data can be taken from (https://www.kaggle.com/sudalairajkumar/indian-startup-funding) or can be scraped from (https://www.trak.in) to get latest data
# 
# A. How does the funding ecosystem change with time?
# 
# B. Do cities play a major role in funding?
# 
# C. Which industries are favored by investors for funding?
# 
# D. Who are the important investors in the Indian Ecosystem?
# 
# E. How much funds does startups generally get in India?
# 

# ## Importing required libraries
# 
# 1. **Pandas:** For loading data from csv into dataframe
# 2. **datetime:** For converting date stored in string formate into datetime format to do analysis based on date, month, quarter, year
# 3. **re:** For extrarcting value from string using pattern
# 4. **matplotlib and seaborn:** For data visualization

# In[ ]:


import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# 1. Initial data exploration using shape, describe() and info()

file_path = "/kaggle/input/indian-startup-funding/startup_funding.csv"
data = pd.read_csv(file_path)
data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


# 2. The data types of data in few columns are not correct to do statistical analysis, hence it needs to be fixed using dtype, astype()

data.rename(columns={"Date dd/mm/yyyy": "Date", 'Startup Name': 'StartupName', 'Industry Vertical':'IndustryVertical',
       'SubVertical': 'SubVertical', 'City  Location': 'CityLocation', 'Investors Name': 'InvestorsName', 'InvestmentnType':'InvestmentnType',
       'Amount in USD':'AmountUSD', 'Remarks':'Remarks'}, inplace=True)
data.head()


# In[ ]:


# 3. Few values 'Date' column are not in correcdt date format checking those rows
data[~data.Date.str.contains('(\d{2})[/](\d{2})[/](\d{4})')]


# In[ ]:


#incorrect_date = data[~data.Date.str.contains('(\d{2})[/](\d{2})[/](\d{4})')].index # [192, 2571, 2606, 2775, 2776, 2831, 3011, 3029]
data.loc[ 192, 'Date'] = '05/07/2018' 
data.loc[2571, 'Date'] = '01/07/2015' 
data.loc[2606, 'Date'] = '10/07/2015'
data.loc[2775, 'Date'] = '12/05/2015' 
data.loc[2776, 'Date'] = '12/05/2015'
data.loc[2831, 'Date'] = '13/04/2015' 
data.loc[3011, 'Date'] = '15/01/2015'
data.loc[3029, 'Date'] = '22/01/2015'


# In[ ]:


# 4. Extracting date, month, year from string values in 'Date' column

date_expand = data['Date'].str.extract(r'(\d{2})/?(\d{2})/?(\d{4})')
data['Year'] = date_expand[2]
data['Month'] = date_expand[1]
data['NewDate'] = date_expand[0]+'/'+date_expand[1]+'/'+date_expand[2]
data.head()


# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])#['Date']
data.head()


# In[ ]:


# 5. Converting datatype of values in 'AmountUSD' column from string to float. Marking Undisclosed values to 'nan' and then converting into float type

data.loc[data['AmountUSD'].isin(['undisclosed', 'unknown', 'Undisclosed']), 'AmountUSD'] = 'nan'

data['AmountUSD'] = data['AmountUSD'].astype(str)
data['NewAmountUSD'] = data['AmountUSD'].apply(lambda x : re.sub("[^0-9]", "", x))
data.loc[data['NewAmountUSD']=='', 'NewAmountUSD'] = 0 #'nan' # replace with average of funding provided that months 
data['NewAmountUSD'] = data['NewAmountUSD'].astype(float)
data.head()


# In[ ]:


# 6. Cleaning column 'CityLocation'

data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Noida', '\\xc2\\xa0Noida']), 'CityLocation'] = 'Noida'
data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Bangalore', '\\xc2\\xa0Bangalore', 'Bangalore']), 'CityLocation'] = 'Bengaluru'
data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0New Delhi', '\\xc2\\xa0New Delhi']), 'CityLocation'] = 'New Delhi'
data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Gurgaon', 'Gurugram']), 'CityLocation'] = 'Gurgaon'
data.loc[data['CityLocation'].isin(['\\\\xc2\\\\xa0Mumbai', '\\xc2\\xa0Mumbai']), 'CityLocation'] = 'Mumbai'
# len(data['CityLocation'].unique())


# In[ ]:


# 7. Cleanning column 'IndustryVertical'

data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0News Aggregator mobile app", 'IndustryVertical'] = 'News Aggregator mobile app'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Online Jewellery Store", 'IndustryVertical'] = 'Online Jewellery Store'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Fashion Info Aggregator App", 'IndustryVertical'] = 'Fashion Info Aggregator App'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Online Study Notes Marketplace", 'IndustryVertical'] = 'Online Study Notes Marketplace'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Warranty Programs Service Administration", 'IndustryVertical'] = 'Warranty Programs Service Administration'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Pre-School Chain", 'IndustryVertical'] = 'Pre-School Chain'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Premium Loyalty Rewards Point Management", 'IndustryVertical'] = 'Premium Loyalty Rewards Point Management'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Contact Center Software Platform", 'IndustryVertical'] = 'Contact Center Software Platform'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Casual Dining restaurant Chain", 'IndustryVertical'] = 'Casual Dining restaurant Chain'
data.loc[data['IndustryVertical'] == "\\\\xc2\\\\xa0Online Grocery Delivery", 'IndustryVertical'] = 'Online Grocery Delivery'
data.loc[data['IndustryVertical'] == "Online home d\\\\xc3\\\\xa9cor marketplace", 'IndustryVertical'] = 'Online home decor marketplace'
data.loc[data['IndustryVertical'].isin(["ECommerce", "E-Commerce", "E-commerce", "Ecommerce"]), 'IndustryVertical'] = 'eCommerce'
data.loc[data['IndustryVertical'].isin(["Fin-Tech"]), 'IndustryVertical'] = 'FinTech'


# In[ ]:


# 8. Cleanning column 'InvestorsName'

data.loc[data['InvestorsName'].isin(['Undisclosed investors', 'Undisclosed', 'undisclosed investors', 'Undisclosed Investor', 'Undisclosed investors']), 'InvestorsName'] = 'Undisclosed Investors'
data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0Tiger Global", 'InvestorsName'] = 'Tiger Global'
data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0IndianIdeas.com", 'InvestorsName'] = 'IndianIdeas'
data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0IvyCap Ventures, Accel Partners, Dragoneer Investment Group", 'InvestorsName'] = 'IvyCap Ventures, Accel Partners, Dragoneer Investment Group'
data.loc[data['InvestorsName'] == "\\\\xc2\\\\xa0Goldman Sachs", 'InvestorsName'] = 'Goldman Sachs'


# ## Starting EDA now!

# In[ ]:


startup_data = data[['Date', 'Year', 'Month', 'StartupName', 'IndustryVertical', 'SubVertical', 'CityLocation', 'InvestorsName', 'InvestmentnType', 'NewAmountUSD']]
startup_data['Date'] = pd.to_datetime(startup_data.Date)
startup_data.set_index('Date', inplace=True)
startup_data.head()


# # A. How does the funding ecosystem change with time?

# In[ ]:


funding_count_yr = pd.DataFrame(startup_data['Year'].value_counts())
funding_count_yr.rename(columns={"Year":"Number of Fundings"}, inplace=True)
funding_count_yr


# In[ ]:


funding_count_qtr = pd.DataFrame(data=startup_data['Year'].resample('QS').count())
funding_count_qtr.rename(columns={'Year':'Number of Fundings(Qtr)'}, inplace=True)
funding_count_qtr['QtrMonth'] = ['2015-1', '2015-4', '2015-7', '2015-10', '2016-1', '2016-4', '2016-7', '2016-10', '2017-1', '2017-4', '2017-7', '2017-10', '2018-1', '2018-4', '2018-7', '2018-10', '2019-1', '2019-4', '2019-7', '2019-10', '2020-1', '2020-4', '2020-7', '2020-10']
funding_count_qtr.head()


# In[ ]:


funding_total_yr = pd.DataFrame(startup_data.groupby(by=['Year'])['NewAmountUSD'].sum())
funding_total_yr.rename(columns={"NewAmountUSD":"Total Funding(USD-Bn)"}, inplace=True)
funding_total_yr = funding_total_yr.sort_values(by='Total Funding(USD-Bn)', ascending=False)
funding_total_yr


# In[ ]:


funding_total_qtr = pd.DataFrame(data=startup_data['NewAmountUSD'].resample('QS').sum())
funding_total_qtr.rename(columns={'NewAmountUSD':'Total Fundings(Qtr USD-Bn)'}, inplace=True)
funding_total_qtr['QtrMonth'] = ['2015-1', '2015-4', '2015-7', '2015-10', '2016-1', '2016-4', '2016-7', '2016-10', '2017-1', '2017-4', '2017-7', '2017-10', '2018-1', '2018-4', '2018-7', '2018-10', '2019-1', '2019-4', '2019-7', '2019-10', '2020-1', '2020-4', '2020-7', '2020-10']
funding_total_qtr.head()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,8))

sns.barplot(x=funding_count_yr.index, y=funding_count_yr['Number of Fundings'], data=funding_count_yr, ax=axes[0,0], orient='v')
sns.barplot(x=funding_count_qtr.index, y=funding_count_qtr['Number of Fundings(Qtr)'], data=funding_count_qtr, ax=axes[0,1], orient='v').set_xticklabels(rotation=90, labels=funding_total_qtr['QtrMonth'])

sns.barplot(x=funding_total_yr.index, y=funding_total_yr['Total Funding(USD-Bn)'], data=funding_total_yr, ax=axes[1,0], orient='v')
sns.barplot(x=funding_total_qtr.index, y=funding_total_qtr['Total Fundings(Qtr USD-Bn)'], data=funding_total_qtr, ax=axes[1,1], orient='v').set_xticklabels(rotation=90, labels=funding_total_qtr['QtrMonth'])

fig.tight_layout(pad=3)
plt.show()


# ## Exploring number and total fundings received by indian startups:
# 
# ### **From above exploration it seems that the ***number of fundings*** made in 2016 was highest(initial quarters in 2016) followed by 2015 but it gradually started decreasing.**
# 
# -> Lot of startups got the funding in the year 2016 and 2015, this might be due to *Startup India, an initiative of the Government of India* declared in August 2015. 
# -> Giving a boost to new entrepreneur getting funds and transforming ideas into reality. Year 2015 and 2016 has spread positivity in indian startup investor market!!
# 
# ### **Getting highest number of fundings in 2016 does not mean the total funds raised was highest in 2016 (which was only 0.38 Billion).** 
# 
# -> The Total amount invested in startups was highest in 2015(majorly 3rd Quarter of 2015 when policy was declared by the Gov.) with 1.6 billion USD followed by in 2019 ~ 1.2 billion USD.
# 
# ### **It seems number of fundings slowed down after 2016 (one of the reason can be Demonatization which affected many industries and markets). **
# 
# -> We have now entered in 2020 with lowest funding raised compared to past data. There seems some issue with domestic investors, why fundings are going down?
# 
# ### The best is yet to come for the investors and the startups looking for fundings.. Hope upcoming days and policies are positive to help them to build amazing projects..

# ## B. Do cities play a major role in funding?
# We all know that the Bangluru is startup hub of India, lets see if the data is supporting the same or an other city is emerging in this race?

# In[ ]:


fundings_count_city = pd.DataFrame(startup_data['CityLocation'].value_counts().sort_values(ascending=False)[:10])
fundings_count_city.rename(columns={'CityLocation':'Number of Fundings by City'}, inplace=True)
fundings_count_city.head()


# In[ ]:


funding_total_city = pd.DataFrame(startup_data.groupby('CityLocation')['NewAmountUSD'].sum()).sort_values(by="NewAmountUSD", ascending=False)[:15]
funding_total_city.rename(columns={'NewAmountUSD':'Total Funding by City(USD-Bn)'}, inplace=True)
funding_total_city.head()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

sns.barplot(x=fundings_count_city.index, y=fundings_count_city['Number of Fundings by City'], data=fundings_count_city, ax=axes[0])
sns.barplot(x=funding_total_city.index, y=funding_total_city['Total Funding by City(USD-Bn)'], data=funding_total_city, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_city.index)

fig.tight_layout(pad=0.5)
plt.show()


# ## We were correct, Bangaluru it is.. the Sillicon Valley of India. 
# 
# -> There is a saying that anyone who comes to Mumbai with willpower will find his/her way to earn and live here. It seems true for statups in Bangaluru as well, the city is providing good atmospheare and is encoraging budding ideas to come to reality. 
# 
# -> Bangaluru has major share of its own by helping Indian economy i.e. by providing employment. 
# 
# -> Total fundings received by Bangaluru is highest at about 2.19 Billion and It drastically drops with Mumbai and NCR regions(Delhi, Gurgaon and Noida) followed by Chennai, Pune, Hydrabad. 
# 
# 
# ## **This explains why is their salary difference in Bangaluru and other cities!! :) Bangaluru take me.. mold me.. teach me.. let me exlore your awsomeness..** :P
# 

# ## C. Which industries are favored by investors for funding?

# In[ ]:


fundings_count_industry = pd.DataFrame(startup_data['IndustryVertical'].value_counts().sort_values(ascending=False))[:15]
fundings_count_industry.rename(columns={'IndustryVertical':'Number of Fundings by Industry'}, inplace=True)
fundings_count_industry.head()


# In[ ]:


funding_total_industry = pd.DataFrame(startup_data.groupby('IndustryVertical')['NewAmountUSD'].sum()).sort_values(by="NewAmountUSD", ascending=False)[:15]
funding_total_industry.rename(columns={'NewAmountUSD':'Total Funding by Industry(USD-Bn)'}, inplace=True)
funding_total_industry.head()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))

sns.barplot(x=fundings_count_industry.index, y=fundings_count_industry['Number of Fundings by Industry'], data=fundings_count_industry, ax=axes[0]).set_xticklabels(rotation=90, labels=fundings_count_industry.index)
sns.barplot(x=funding_total_industry.index, y=funding_total_industry['Total Funding by Industry(USD-Bn)'], data=funding_total_industry, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_industry.index)

fig.tight_layout(pad=1)
#plt.xticks(rotation=90)
plt.show()


# ## 941 statups have got the fundings for **Consumer Internet Industry**, that seems perfectly fine as the users cunsumption growing, and we are part of it. 
# 
# -> Contribution in research and projects in Technology, eCommerce and Helthcare has boosted fundings and new automations are coming in to help the cosumers.
# 
# -> The total amount of fundings received for eCommerce is 0.87 billion and followed by Consumer Internet(0.6 billion), Trasportation(0.39 billion) and Technology(0.22 Billion)
# 
# ### Many new startups are coming in the market to serve us like online Jewellery, news aggregation apps, grocery delivery, online pharmacy, educational content provider, electric scooter manufactoring. We are definatly growing and catching up with western world.

# ## D. Who are the important investors in the Indian Ecosystem?

# In[ ]:


funding_count_investor = pd.DataFrame(startup_data['InvestorsName'].value_counts()).sort_values(by='InvestorsName', ascending=False)[:10]
funding_count_investor.rename(columns={'InvestorsName': 'Number of Investments by Investor'}, inplace=True)
funding_count_investor.drop(funding_count_investor[funding_count_investor.index == 'Undisclosed Investors'].index, inplace=True)
funding_count_investor.head()


# In[ ]:


funding_total_investor = pd.DataFrame(startup_data.groupby(['InvestorsName'])['NewAmountUSD'].sum()).sort_values(by="NewAmountUSD", ascending=False)[:15]
funding_total_investor.rename(columns={'NewAmountUSD':'Total Funding by Investor(USD-Bn)'}, inplace=True)
funding_total_investor.head()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))

sns.barplot(x=funding_count_investor.index, y=funding_count_investor['Number of Investments by Investor'], data=funding_count_investor, ax=axes[0]).set_xticklabels(rotation=90, labels=funding_count_investor.index)
sns.barplot(x=funding_total_investor.index, y=funding_total_investor['Total Funding by Investor(USD-Bn)'], data=funding_total_investor, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_investor.index)

fig.tight_layout(pad=1)
plt.show()


# In[ ]:


startup_data[startup_data['InvestorsName'].isin(['Westbridge Capital', 'Softbank'])]


# ## A one-man venture capital fund is funding Indian startups giving lot of credibility and future. 
# 
# -> Except number of "Undisclosed Investors" the second rank comes to **Sir Ratan Tata**, Number of startups he has funded are 25 and increasing. This is followed by Indian Angel Network(23), Kalaari Capital(16) etc.
# 
# -> Total amount invested 0.39 billion USD by Westbridge Capital	is the highest in 'Rapido Bike Taxi' and Softbank has invested 0.29 billion USD in Flipkart

# ## E.How much funds does startups generally get in India?

# In[ ]:


funding_count_company = pd.DataFrame(startup_data['StartupName'].value_counts()).sort_values(by='StartupName', ascending=False)[:15]
funding_count_company.rename(columns={'StartupName': 'Number of Investments by Investor'}, inplace=True)
funding_count_company.head()


# In[ ]:


funding_total_company = pd.DataFrame(startup_data.groupby('StartupName')['NewAmountUSD'].sum()).sort_values(by='NewAmountUSD', ascending=False)[:15]
funding_total_company.rename(columns={'NewAmountUSD': "Total amount Raised by Startup (USD-Bn)"}, inplace=True)
funding_total_company.head()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))

sns.barplot(x=funding_count_company.index, y=funding_count_company['Number of Investments by Investor'], data=funding_count_company, ax=axes[0]).set_xticklabels(rotation=90, labels=funding_count_company.index)
sns.barplot(x=funding_total_company.index, y=funding_total_company['Total amount Raised by Startup (USD-Bn)'], data=funding_total_company, ax=axes[1]).set_xticklabels(rotation=90, labels=funding_total_company.index)

fig.tight_layout(pad=1)
plt.show()


# ### Swiggy, Ola Cabs, Paytm, Nykaa etc has got lot of fundings these years. This due to their constant innovation in surving cunsumer with AI.
# 
# -> Total funding received by Flipkart is highest 0.4 billion USD followed by Rapido Bike Taxi with 0.39 billion USD.

# In[ ]:


plt.figure(figsize=(25,8))
sns.distplot(startup_data.loc[startup_data['NewAmountUSD']<=10000000.0, 'NewAmountUSD'])
plt.show()


# In[ ]:


funding_average = startup_data['NewAmountUSD'].mean()
funding_meadian = startup_data['NewAmountUSD'].median()
print(funding_average, funding_meadian)


# ### Average fundings received is 15 million USD and most of the companies get 5 lac. Indian startup market is helping new innovations to come to live and encouraging entrepreneurs. 
# 
# -> Ending my EDA on this data, got to learn so much about startupd and investor market in India, hope to see more growth in investers.
# 
