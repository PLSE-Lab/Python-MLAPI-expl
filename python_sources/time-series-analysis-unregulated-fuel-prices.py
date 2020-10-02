#!/usr/bin/env python
# coding: utf-8

# # TIME SERIES ANALYSIS, COMPARING AGGREGATED UNREGULATED FUEL PRICES AMONGST LEADING SOUTH AFRICAN OIL COMPANIES**

# ### Introduction
# 
# The dataset contains individual transactions made on fleet fuel cards (garage cards) belonging to a private company's fleet of vehicles, mainly located around South Africa. South Africa has a watchful eye on their fuel prices, typically due to it's volatility. 
# 
# Having this disposable data, there is a motivation to aggregate, or atleast get an idea, how competitive unregulated fuel prices are amongst leading oil companies operating in South Africa. 
# 
# At the end of this time-series analysis, an answer should be available to "On average, what is the cheapest oil company to buy diesel from?"
# 
# ### Breakdown
# * Data cleaning and exploration
# * Plotting the price per litre for unregulated fuel
# * Getting aggregates
# * Conclusions
# 
# ------------

# 1) Data Cleaning and Exploration

# In[ ]:


import pandas as pd
import seaborn as sns
df = pd.read_excel('../input/Fuel_Transactions_by_Oil_Company.xlsx')
df.info()


# In[ ]:


df['Brand Description'].value_counts()


# In[ ]:


df['Fuel Type'].value_counts() #two types of fuel


# In[ ]:


df['PPL'] = df['Amount'] / df['Quantity'] #lets bring in the price per litre
#remove price regulated fuel types in South Africa from the dataframe, in this case petrol
df = df[df['Fuel Type'] != 'Petrol']


# In[ ]:


import pandas_profiling
df.profile_report()


# In[ ]:


plot = sns.lineplot(x='Transaction Date', y='PPL', data=df, hue='Brand Description') #plot for interest


# Since this is a time-series analysis, we'll need to address the logical known influencing factors on PPL: location and time. 
# 
# With the influencing factors in mind, we will only sample groups with observations taken in the same month and same location (merchant town):

# In[ ]:


df = df[df.duplicated(subset=['Merchant Town'], keep=False)] #address merchant towns
df = df.sort_values(by=['Brand Description', 'Transaction Date']) #address time
df['Date Diff'] = df['Transaction Date'].diff() 
pd.to_datetime(df['Date Diff'])
df['total_days_td'] = df['Date Diff'] / pd.to_timedelta(1, unit='D')
df = df[df['total_days_td'] >= 0] #remove brand description group days difference overlap
df = df.groupby('Brand Description').filter(lambda g: (g.total_days_td < 30).all()) #filter brands not used within 30 days
df['Brand Description'].unique() #lets check the sample


# In[ ]:


df = df[df['Brand Description'] != 'OTHER/ANDER'] #Remove unidentified fuel brands
df = df[df['Brand Description'] != 'BRENT OIL'] #Remove unidentified fuel brands
df = df[df['Brand Description'] != 'SERVICES AND MAINTENANCE'] #Remove unidentified brands
df['Brand Description'].unique() #our resultant sample groups


# 2) Creating plots

# In[ ]:


plot = sns.lineplot(x='Transaction Date', y='PPL', data=df, hue='Brand Description')


# We can plot the 30 day moving average PPL for each brand for interest and see if there is more of an identifiable trend

# In[ ]:


df['MA'] = df.groupby('Brand Description')['PPL'].transform(lambda x: x.rolling(30, 1).mean())
plot = sns.lineplot(x='Month', y='MA', data=df, hue='Brand Description')


# 3) Calculating Aggregates

# Lastly, we can plot the average PPL per brand over the time period - answering our main question

# In[ ]:


plot = sns.barplot(x='Brand Description', y='PPL', data=df)


# In[ ]:


df.groupby('Brand Description')['PPL'].mean().sort_values()


# 4) Conclusion
# As we can see, the unregulated fuel PPL (diesel) per oil company has been plotted and aggregated. We notice that there are slight differences in prices in oil companies. It is worth reiterating that the location of a merchants fuel station is correlating to the PPL - this is due to the logistical costs of getting the fuel from the refineries to the different locations. Thus, aggregated functions is the only method to compare the unregulated fuel PPLs.
