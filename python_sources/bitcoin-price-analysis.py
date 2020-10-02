#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from datetime import datetime
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# - Preprocessing bitcoin_price.csv to get the date in right format

# In[4]:


bit_price_df = pd.read_csv('../input/bitcoin_price.csv')
for i, row in bit_price_df.iterrows():
    dt = datetime.strptime(row['Date'], '%b %d, %Y')        
    dt = dt.strftime('%Y-%m-%d')
    row['Date'] = dt
    bit_price_df.set_value(i,'Date',dt)


# In[6]:


bit_price_df.head()


# - Preprocessing bitcoin_dataset.csv to get the date in right format

# In[7]:


bitcoin_dataset_df = pd.read_csv('../input/bitcoin_dataset.csv')

for i, row in bitcoin_dataset_df.iterrows():
    dt = datetime.strptime(row['Date'], '%Y-%m-%d 00:00:00')        
    dt = dt.strftime('%Y-%m-%d')
    row['Date'] = dt
    bitcoin_dataset_df.set_value(i,'Date',dt)


# In[8]:


bitcoin_dataset_df.head()


# - Joining the two dfs on 'Date' column
# 

# In[13]:


joined_data = bitcoin_dataset_df.merge(bit_price_df, on='Date')


# In[14]:


joined_data.head()


# ### Let Data tell the story
# - Date vs price
# - day of week analysis
# - seasonal trend in price
# - check bitcoin price vs block size
# - check price correlation with all other variables in that dataset and see which one are correlated
# - try to find out the if you can see any relations
# 

# In[15]:


sns.distplot(joined_data['Close'], kde=False, label='closing price') #default bins using Freedman-Diaconis rule.
#sns.distplot(joined_data['Open'], kde=False, label='Open price') #default bins using Freedman-Diaconis rule.
#sns.distplot(joined_data['High'], kde=False, label='High price') #default bins using Freedman-Diaconis rule.
plt.title("Distribution of closing price of Bitcoin")
plt.legend(loc='best')
plt.show()


# In[16]:


import datetime

fig, ax = plt.subplots(figsize=(12,8))
x3 = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in joined_data.Date]

#joined_data
joined_data['moving_avg'] =  joined_data['Close'].rolling(window=30).mean()
#print(joined_data)


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.plot(x3, joined_data.Close, label='closing price')
plt.plot(x3, joined_data.moving_avg, color='red', label='moving average(close price)')
plt.gcf().autofmt_xdate()
plt.xlabel("Date", fontsize=15)
plt.ylabel("Bitcoin Price", fontsize=15)
plt.title("Bitcoin Price overtime", fontsize=20)
plt.legend(loc='best')
plt.show()


# The plot shows a huge surge in price of bitcoin in 2017 
# 
# The price is increased from 1000 to 3000 dollar from earlier 2017 till July 2017 
# and then from 3000 to 7000 dollars from July to November.

# ## Day of week analysis
# 

# In[31]:


joined_data['weekday'] = pd.to_datetime(joined_data['Date']).dt.weekday_name
week_data = joined_data.groupby(['weekday'], as_index=False)['Close'].agg({'mean': 'mean'})
day_of_week = pd.DataFrame(data=week_data)


plt.plot(figsize=(12,8))
plt.title('Day of Week Analysis')

my_xticks = np.array(day_of_week.weekday)
plt.xticks(range(len(week_data['mean'])), my_xticks)
plt.plot(range(len(week_data['mean'])), week_data['mean'])


# - The plot shows that Saturday and Tuesday see the high bitcoin price while Wednesday has the lowest price. 

# - But just looking at the average might be misleading. Let's look at spread of data per year.

# In[34]:


joined_data['year'] = pd.to_datetime(joined_data['Date']).dt.year
joined_data['month'] = pd.to_datetime(joined_data['Date']).dt.month
joined_data['weekday'] = pd.to_datetime(joined_data['Date']).dt.weekday_name

#print(joined_data.weekday)

#week_data=joined_data.groupby(['weekday'])['Close'].mean()

mean_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'mean': 'mean'})
std_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'std': np.std})
min_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'min': np.min})
max_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'max': np.max})
median_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'median': np.median})



week_data = pd.concat([mean_df, std_df['std'], min_df['min'],max_df['max'], median_df['median']], axis=1)
week_data['var_coeff'] = std_df['std'] / mean_df['mean']

week_data.head(10)


# In[36]:


years = [2013,2014,2015,2016,2017]


fig, ax = plt.subplots(len(years),4,sharex=True, sharey=False ,figsize=(12,12))
fig.suptitle('Day of Week Analysis')

for i, year in enumerate(years):
    holder = week_data[week_data['year']==year]
    #print(holder)
    
    my_xticks = np.array(day_of_week.weekday)
    plt.xticks(range(len(holder['mean'])), my_xticks, rotation=90)
    
    ax[i][0].plot(range(len(holder['mean'])), holder['mean'])
   # ax[i][0].set_ylim(min(holder['mean']), max(holder['mean']))
    

    ax[i][1].errorbar(
    range(len(holder['mean'])),     # X
    holder['mean'],    # Y
    yerr=holder['std'],        # Y-errors
      # format line like for plot()
    linewidth=3,   # width of plot line
    elinewidth=1,# width of error bar line
    ecolor='r',    # color of error bar
    capsize=4,     # cap length for error bar
    capthick=2,  # cap thickness for error bar
    )
    

    ax[i][2].plot(range(len(holder['mean'])), holder['mean'])
    #ax[i][2].set_ylim(abs(max(holder['mean']) - max(holder['std'])), abs(max(holder['mean']) + max(holder['std'])))
    
    ax[i][3].errorbar(
    range(len(holder['mean'])),     # X
    holder['mean'],    # Y
    yerr=holder['std'],        # Y-errors
      # format line like for plot()
    linewidth=3,   # width of plot line
    elinewidth=1,# width of error bar line
    ecolor='r',    # color of error bar
    capsize=4,     # cap length for error bar
    capthick=2,  # cap thickness for error bar
    )
    
    
    ax[i][2].set_ylim(0, 4500) #these values are set by experimenting with 'sharey' attribute in subplots() function
                               #the goal is to compare the mean for all years relative to bitcoin increased price now
    ax[i][3].set_ylim(0, 4500)

    #ax[i].set_xlabel("Year"+" "+str(year),fontsize=10)
    #ax[i].set_ylabel("Bitcoin Price",fontsize=10)
    ax[i][0].set_title(year)
    ax[i][1].set_title(year)
    ax[i][2].set_title(year)   
    ax[i][3].set_title(year)  

for tick in ax[i][0].get_xticklabels():
    tick.set_rotation(90)
for tick in ax[i][1].get_xticklabels():
    tick.set_rotation(90)
for tick in ax[i][2].get_xticklabels():
    tick.set_rotation(90)


#plt.savefig('yearly_dayofweek.png')
plt.show()


# 
# The price trend for weekdays is not very clear as it was not in 5 years data plot. 
# But we can still see that:
# - In three of five years, Wednesdays see price drops and Mondays see price surge.
# - The trend is alternating years is comparable as well. Forexample year 2013, 2015 and 2017 somehow follow the similar trend.
# 
# If we look at the mean for day-of-week individually, it seems mean is fluctuating and there is no visible pattern as we saw before. 
# 
# may be price is not a good indicator. How about **difference** in price?

# ## See spread over the years to see consistency 

# In[39]:



years = [2013,2014,2015,2016,2017]

fig,ax = plt.subplots(figsize=(12, 8))  #plt.subplots(figsize=(12,8))
fig.suptitle('Month of year - Price Spread')
#fig.subplots_adjust(hspace=.5,wspace=0.7)


for i, year in enumerate(years):
    holder = year_month_data[year_month_data['year']==year]
    #print(holder)
    ax.plot(range(len(holder['std'])), holder['std'], label=str(year))
    ax.set_xlabel("Year",fontsize=10)
    ax.set_ylabel("Price Spread",fontsize=10)
    #ax[i].title("Month of year Price"+" "+str(year), fontsize=10)
    
    
my_xticks = np.array(holder.month)
plt.xticks(range(len(holder['std'])), my_xticks)#Set label location
plt.legend(loc='best')
#plt.savefig('btc_stability.png')
plt.show()


# - This shows the point that BTC price is volatile rather the most unstable in 2017.

# # BTC price correlation with some other variables

# I chose these variables based on my understanding of bitcoin. To me, these variables might affect the bitcoin price.
# - Close: closing price of bitcoin
# - btc_market_cap: bitcoin market capitilization for the particular date. Market cap = Total bitcoin volume * price.
# - btc_n_transactions_per_block: How many transactions took place per block on a particular date
# - abs_btc_count: bitcoin count till date
# - btc_hash_rate: Hash rate applied to mine the bitcoins per date
# - btc_difficulty: measure of difficulty to mine. The more difficult, the less coins should be mined.
# - btc_cost_per_transaction: avg cost of transaction
# - btc_n_transactions:

# In[41]:


selected_col = joined_data[['Close','btc_market_cap',
                            'btc_avg_block_size',
                            'btc_n_transactions_per_block',
                            'btc_hash_rate',
                            'btc_difficulty',
                            'btc_cost_per_transaction',
                            'btc_n_transactions']]

selected_col.head()
corrmat = selected_col.corr(method='pearson')

columns = ['Close']
my_corrmat = corrmat.copy()
mask = my_corrmat.columns.isin(columns)
my_corrmat.loc[:, ~mask] = 0
#print(my_corrmat)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(my_corrmat, annot=False, fmt="f", cmap="Blues") #vmax=1., square=True)
plt.title("Correlation Between Price and other factors", fontsize=15)
#plt.savefig('variablecorrelation.png', bbox_inches='tight')
plt.show()


# Since, We are concerned with the relationship between bitcoin and other variables, I have made other values 0 for the sake of clarity.
# 
# The map shows that BTC price is most correlated with BTC market capitilization, BTC hash rate and BTC_difficulty
# 

# In[ ]:




