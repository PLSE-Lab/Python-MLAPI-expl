#!/usr/bin/env python
# coding: utf-8

# # Two Sigma Outlier investigation
# 
# The two sigma dataset includes a fair amount of outliers. I will investigate the validity of the outliers below and document what I find. I will start with the target variable and possibly move on to other variables. Once I post my initial kernal, I will try to build on my work using and siting any useful techniques from other kernals. 
# 
# 
# # Load Librarys and Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from kaggle.competitions import twosigmanews # competition data
import matplotlib.pyplot as plt # graphs

env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data() # load training data


# In[ ]:


# Summary of raw target variable.
target = market_train['returnsOpenNextMktres10']
print(f"Data for the target variable is centered at {target.mean():.3f} with a standard deviation (SD) of {target.std():.3f}. The mean \nseems reasonable, but when veiwing a histogram of the data, it is clear that outliers are driving the \nSD. Out of {len(target)} total values, the biggest negative target value is {target.min():.0f} and there are {sum(target < -0.5)} values under -0.5; the largest posative number is {target.max():.0f} and there are {sum(market_train['returnsOpenNextMktres10'] > 1)} values over 0.5. The histogram below shows the shape of target variable for values over -0.5 and under 0.5 with the data 'clipped' using pandas clip method at -0.5 and 0.5. The outliers can be seen graphically (with a good monitor) at -0.5 and 0.5 on the x-axis.")
    
target.clip(-.5,0.5).hist(bins=400, figsize=(7,5))
plt.title('Target Variable Distribution', size=14)
plt.xlim(-0.6, 0.6)
plt.show()


# ## Investigation
# 
# I would like to find where outliers are, if they are valid readings, and how to remove them and any other bad data I can identify. The easiest way to get rid of the outliers will be to set high and low limits for the target variable to subset the data frame. From the graph above it looks like -0.5 and positive 0.5 are reasonable values to use as limits. 
# 
# I will start by identifying the largest outliers for the target variable, then I will take a look at them against the current open value as it is the value used to calculate them. I will also take a look at a few other summarizations of the data. I wrote a short function using two matplotlib graphs to illustrate the root cause of the bad target values. Unhide the next cell to view the check_bad_opens function I will use to check the extreme values.
# 

# In[ ]:


# I made a graph to look at the high target values against stock open price

def check_bad_opens(asset_code, df=market_train, show_graph=True):
    """ Expects market_train data frame format. Returns pyplot graphing opeject or plots graph.
    
        Parameters
        ----------
        asset_code (string): assetCode from market_trian df to plot
        df (pandas.DataFrame): market_train df from two sigmas
        show_graph (boolean): can be set to False to make further modifications
    """
    
    dat = df[df['assetCode'] == asset_code]
    name_for_title = dat['assetName'][0]
    dat.index = dat['time']
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15,8))
    dat['open'].plot(ax=ax1)
    ax1.grid()
    dat['returnsOpenNextMktres10'].plot(ax=ax2)
    plt.grid()
    ax1.set_title('Daily Open Values for ' + name_for_title, size=18)
    ax2.set_title('Adjusted Next 10 Day Return ' + name_for_title, size=18)
    plt.tight_layout()
    if show_graph:
        plt.show()
    else:
        return ax1, ax2


# I will start by looking at the extreem outliers. The figure below illustrates the cause of the extreem values. A bad (close to zero) open value caused the calculated returns to go 'crazy' for a while. The next graph takes a closer look at the periods before and after the bad open value.  

# In[ ]:


# Check for extreem outliers and take a look at them using check_bad_open
outlier_dat = market_train[ (market_train['returnsOpenNextMktres10'] > 1000) | (market_train['returnsOpenNextMktres10'] < -1000) ]
outlier_dat['assetCode'].value_counts()
market_train['time'] = pd.to_datetime(market_train['time'])


# In[ ]:


check_bad_opens('ATPG.O')


# In[ ]:


ax1, ax2 = check_bad_opens('ATPG.O', show_graph=False)
ax1.set_xlim(pd.datetime(2007, 10,1), pd.datetime(2008, 1,1))
ax2.set_xlim(pd.datetime(2007, 10,1), pd.datetime(2008, 1,1))
plt.tight_layout()
plt.show()


# The above graph shows that the 10 day return (target) goes out of range for around a month due to the bad open value. I will try to find all of the bad open values in the next cell. 

# In[ ]:


## now that it looks like bad open values are driving the bad return values. I will take a look at the distribution of the 
## open values. 

plt.hist(market_train['open'].clip(0,400), bins=400)
plt.title('Two Sigma Open Values for Period of Record')
plt.show()


# In[ ]:


## I can see from the histogram that there are high and low extreem open values. I will assume that opens below 0.05 and 
## above 999 are bad

bad_open_df = market_train[ (market_train['open'] < 0.2) | # or 
                            (market_train['open'] > 900) ]


# In[ ]:


bad_open_df['assetCode'].value_counts()


# In[ ]:


## It looks like PCLN and PGN have a lot of bad open values, so I will take a look at them with my graph.
check_bad_opens('PCLN.O')
check_bad_opens('PGN.N')


# Both of the assets with multiple high and low open values look like they are not bad data. The first 'Booking Holdings' looks like it just had a huge run, and the second looks like a smaller energy asset that has very high volatility. It might be better for training to take the second asset out, but for now I will leave it in. I checked the rest of the values, and they all had visually apparent 'bad open' values.
# 
# Below I will be removing all rows from the training set 50 days before and after a 'bad open'.

# In[ ]:


## get rid of the values discused above...
bad_open_df = bad_open_df[ (bad_open_df['assetCode'] != 'PCLN.O') &
             (bad_open_df['assetCode'] != 'PGN.N')]


# In[ ]:


get_ipython().run_cell_magic('time', '', "bad_ix = list()\n\nfor x in bad_open_df.index:\n    bad_date = market_train.loc[x,'time']\n    min_date = bad_date - pd.DateOffset(days=50)\n    max_date = bad_date + pd.DateOffset(days=50)\n    site_name = market_train.loc[x, 'assetCode']\n    bad_data = market_train[ (market_train['time'] > min_date) & \n                             (market_train['time'] < max_date) &\n                             (market_train['assetCode'] == site_name)]\n    bad_ix = bad_ix + list(bad_data.index)")


# In[ ]:


data_ = market_train.copy() # since I can't reload the training data I will make a copy

data_ = data_.drop(axis=0, index=bad_ix)


# In[ ]:


# Summary of raw target variable.
target = data_['returnsOpenNextMktres10']
print(f"Data for the target variable is centered at {target.mean():.3f} with a standard deviation (SD) of {target.std():.3f}. The mean \nis now closer to zero, but the SD is still high. Out of {len(target)} total values, the biggest negative target value is {target.min():.0f} and there are {sum(target < -0.5)} values under -0.5; the largest posative number is {target.max():.0f} and there are {sum(market_train['returnsOpenNextMktres10'] > 1)} values over 0.5. The histogram below shows the shape of target variable for values over -0.5 and under 0.5 with the data 'clipped' using pandas clip method at -0.5 and 0.5. The outliers can be seen graphically (with a good monitor) at -0.5 and 0.5 on the x-axis.")
    
target.clip(-.5,0.5).hist(bins=400, figsize=(7,5))
plt.title('Target Variable Distribution', size=14)
plt.xlim(-0.6, 0.6)
plt.show()


# In[ ]:


# identify the sites which still have high target values and check the minimum outputs
sites_list = data_[ (data_['returnsOpenNextMktres10'] < -5) |
                    (data_['returnsOpenNextMktres10'] > 5)]['assetCode'].unique()

min_open = [data_[data_['assetCode'] == x]['open'].min() for x in sites_list]
# max_open = [data_[data_['assetCode'] == x]['open'].max() for x in sites_list] # none of the max opens were that high

min_opens = dict(zip(sites_list, min_open))


# In[ ]:


min_opens 


# In[ ]:


# will use a dict comprehension to get rid of the unwanted values
min_opens = {key:val for (key, val) in min_opens.items() if val < 2}
_ = min_opens.pop('PGN.N') # this was checked above


# In[ ]:


min_opens


# In[ ]:


for asset, min_val in min_opens.items():
    pass


# In[ ]:


# i didn't use the dict for this, just the list and df index
bad_ix = list()

for asset, min_val in min_opens.items():
    bad_date = data_.loc[(data_['assetCode'] == asset) & (data_['open'] == min_val),'time'][0]
    min_date = bad_date - pd.DateOffset(days=50)
    max_date = bad_date + pd.DateOffset(days=50)
    bad_data = data_[ (data_['time'] > min_date) & 
                             (data_['time'] < max_date) &
                             (data_['assetCode'] == asset)]
    bad_ix = bad_ix + list(bad_data.index)


# In[ ]:


data_ = data_.drop(axis=0, index=bad_ix, errors='ignore')


# In[ ]:


# Summary of raw target variable.
target = data_['returnsOpenNextMktres10']
print(f"Data for the target variable is centered at {target.mean():.3f} with a standard deviation (SD) of {target.std():.3f}. The mean \nis now closer to zero, but the SD is still high. Out of {len(target)} total values, the biggest negative target value is {target.min():.0f} and there are {sum(target < -1)} values under -1; the largest posative number is {target.max():.0f} and there are {sum(market_train['returnsOpenNextMktres10'] > 1)} values over 1. The histogram below shows the shape of target variable for values over -0.5 and under 0.5 with the data 'clipped' using pandas clip method at -0.5 and 0.5. The outliers can be seen graphically (with a good monitor) at -0.5 and 0.5 on the x-axis.")
    
target.clip(-.5,0.5).hist(bins=400, figsize=(7,5))
plt.title('Target Variable Distribution', size=14)
plt.xlim(-0.6, 0.6)
plt.show()


# ### to be continued...

# In[ ]:




