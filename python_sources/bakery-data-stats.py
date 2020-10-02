#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def convert_time(b):
    return b[:2]

# Any results you write to the current directory are saved as output.
path=r'../input/BreadBasket_DMS.csv'
df=pd.read_csv(path)


# So here we have time series data in a column Time and Date. For now , I will pay the most attention to time and for my analytical purposes, I'll adopt a quick-and-dirty method to handle time series data using the above convert time function.

# In[ ]:


#we`ll use this for analysis and keep the main dataframe separate, untouched
df_analysis=df.copy()
#Get just the hour of day in the time column
df_analysis['Time']=df['Time'].apply(convert_time)

#check whether it converted successfully.
df_analysis['Time'].head(2)


# Now that we have our data in a format we want, lets do some analytics on it. Lets start off with the most bought items in the bakery.

# In[ ]:


#now lets find the top 5 most bought items in the bakery
df_most_bought=df_analysis.groupby(['Item'])['Item'].count()
df_most_bought=df_most_bought.sort_values(ascending=False)
df_top5=df_most_bought.head(5)
plt.bar(df_top5.index,df_top5)


# Whoa! not a fan of tea. Coffee is holding up good, its almost twice as bread. Hmmm, Well at least they arent buying Pastries enough, That would be a dia-saster . (Presumably funny word play on Diabetes and Disaster). 
# 
# Lets move on to see what times of the day we see the most traffic in the store.

# In[ ]:


#Now lets see what time of the day do we see the most sales
df_grpby=df_analysis.groupby('Time')['Time'].count()

plt.xlabel('hr of the day in 24 hr format')
plt.ylabel('sales made at the hour')
plt.plot(df_grpby.index,
        df_grpby)


# Someone's been busy! It seems to peak between 9 and 10 am . "Cant miss that meeting, Looks like its espresso to go!" . Must be the yuppies, With a coffee in one hand and a newspaper... oh sorry, cell phone in the other.
# 
# Lets check out what days of the year were the busiest for our bakery staff.

# In[ ]:


#lets see which were the busiest days of the year!
#df_grpby=df_analysis.groupby(['Date','Time'])['Item'].count()
df_grpby=df_analysis.groupby(['Date'])['Item'].count()
df_grpby=df_grpby.sort_values(ascending=False)
df_top5_busiest=df_grpby.head(5)
plt.xlabel('dates')
plt.ylabel('transactions that day')
plt.bar(df_top5_busiest.index,
        df_top5_busiest)


# Huge rush of people on 4th of Feb, 2017 . 
# 
# Lets see the time distribution for that day.

# In[ ]:


busiest_day = df_analysis['Date']=='2017-02-04'
df_busiest_day=df_analysis[busiest_day].groupby(['Time'])['Item'].count()
plt.xlabel('Hour of the day')
plt.ylabel('transactions made')
plt.plot(df_busiest_day.index,
df_busiest_day)


# Contrary to what we saw in the overall busiest time graph, This one paints a different picture, It's the busiest somewhere between 11 and 12 pm.
# 
# Stay tuned, there's more to come!
