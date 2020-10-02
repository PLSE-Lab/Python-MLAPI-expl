#!/usr/bin/env python
# coding: utf-8

# # Part 1
# Deals with simple understanding of current Data Design....Check it for simple predictions
# ### Load data
# ### Check data
# ### Plot data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


prop_ds = pd.read_csv('../input/properties_2016.csv',dtype={'hashottuborspa':np.str,'propertycountylandusecode':np.str,'propertyzoningdesc':np.str,'fireplaceflag':np.str,'taxdelinquencyflag':np.str})
train_ds = pd.read_csv('../input/train_2016.csv',parse_dates=['transactiondate'])


# In[ ]:


#Import basic libs
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
color = sns.color_palette()


# In[ ]:


prop_ds.head()


# In[ ]:


train_ds.head()


# In[ ]:


train_ds1 = train_ds
train_ds1['month'] = train_ds1.transactiondate.dt.month


# In[ ]:


train_ds1.head()


# In[ ]:


train_ds1.groupby('month',sort=False)['logerror'].mean()


# In[ ]:


aggregations = {
    'logerror' : 'mean'
}
#train_ds1.groupby('month').agg(aggregations).T.plot(kind='bar',)
#train_ds1.groupby('month')['logerror'].mean().plot(kind='bar') #Gives same result as above
avg_logerr = train_ds1.groupby('month').agg(aggregations)
avg_logerr
avg_logerr.T.plot(kind='bar')


# In[ ]:


aggregations = {
    'parcelid' : 'count'
}
#train_ds1['month'].value_counts()
monthly_sales = train_ds1.groupby('month').agg(aggregations) #It create DF with month as index
monthly_sales = monthly_sales.rename(columns={'parcelid':'NoOfSales'}) #Rename column name from parcelid to NoOfSales
#monthly_sales1.reset_index(level=0,inplace=True) # Move index as column
#monthly_sales1
monthly_sales.T.plot(kind='bar')


# First plot shows that logerror is quite more in month of Oct, Nov and Dec...........
# Second plot shows Sales are less in month of Oct, Nov and Dec...........

# In[ ]:


import pandas as pd
import calendar
avg_logerr
avg_logerr1 = avg_logerr['logerror'].apply(lambda x : x*100)
avg_logerr1

#monthly_sales
#monthly_sales1 = monthly_sales['month'].apply(lambda x : ((x/90811) * 100) - 10)
#monthly_sales1

monthly_sales
monthly_sales1 = monthly_sales['NoOfSales'].apply(lambda x : np.log10(x))
monthly_sales1

compare = pd.DataFrame({'LogError': avg_logerr1[:], 'Sales': monthly_sales1[:]})
compare.reset_index(level=0,inplace=True)
compare['month'] = compare['month'].apply(lambda x: calendar.month_abbr[x])
compare1 = compare.set_index('month')
#del compare['month']
compare1
#month_val = compare.index
#month_val = month_val.apply(lambda x : calendar.month_abbr[x])
#month_val
compare1.plot(kind='bar')

#merge_sales_logerr = pd.merge(avg_logerr,monthly_sales,on = 'month', how='outer')


# In[ ]:


#Find highest popularity Property
aggregation = {
    'parcelid' : 'count'
    }
#prop_ds.groupby('propertylandusetypeid').agg(aggregation)
prop_ds.groupby('propertylandusetypeid').agg(aggregation).plot(kind='bar')


# In[ ]:


prop_ds.groupby('regionidcity').count()


# In[ ]:


#Find number of Properties per County
aggregation = {
       'propertylandusetypeid' : 'count'
    }

prop_ds.groupby(['regionidcounty','propertylandusetypeid']).agg(aggregation).plot(kind='bar')


# In[ ]:


#Devide Properties in each city based on their type per county.

prop_city = prop_ds.groupby(['regionidcounty','regionidcity','propertylandusetypeid'])['propertylandusetypeid'].count()
#prop_city
prop_ds_city = pd.DataFrame(prop_city)
prop_ds_city.rename(columns={'propertylandusetypeid':'Total'},inplace=True)
prop_ds_city.reset_index()
#prop_ds_city.rename(columns={'regionidcounty':'CountyId','regionidcity':'CityId','propertylandusetypeid':'LandTypeId'})
#prop_ds_city.columns.droplevel(0)
#fg = prop_ds_city[prop_ds_city['regionidcounty'] == 1286]
#fg.plot(kind='bar')
#sns.countplot(x="propertylandusetypeid",data=prop_ds_city)
#plt.ylabel('City', fontsize=12)
#plt.xlabel('PropertyType', fontsize=12)
#plt.xticks(rotation='vertical')
#plt.title("CityWise Property", fontsize=15)
#plt.show()


# In[ ]:


#merge_ds = pd.merge(train_ds, prop_ds, on='parcelid', how='left')
#merge_ds.groupby(['regionidcounty','regionidcity','propertylandusetypeid'])['propertylandusetypeid','logerror'].count()

