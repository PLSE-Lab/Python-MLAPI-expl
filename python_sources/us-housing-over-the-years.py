#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df=pd.read_csv('../input/family-households-with-married-couples.csv')
# Any results you write to the current directory are saved as output.


# US Housing over the years is an interesting topic to take up, given the fact how its a principle component of the ***"American Dream"***
# After importing the data , we get to work and see if there are any anomalies in the dataset. An easy way to do so is to usually do a sort. 

# In[ ]:


df["date"]=pd.to_datetime(df["date"])
df2=df.sort_values(by="value")
df2.head(15)


# So the first few rows in the value column seem to be dots "." .
# We can fix that, just filter them out! what other options are there?
# * backfill 
# * forward fill
# * mean
# 
# Each method has its own implications and its best if we just stick to whats in this dataset.

# In[ ]:


gate_incorrect= df2["value"] != "."
df2=df2[gate_incorrect]
df2.head(5)


# There! thats better. Now that the values column is all fixed, lets convert it to a number.

# In[ ]:


df2["value"]=pd.to_numeric(df2["value"])
df2.info()


# Now that all the necessary pre-processing is done, we can get to the fun stuff! analysis! (Just me? ok.....)
# What we're interested in, is the rise and fall of these values over the years. Even if a difference column is added to the dataset, it still wont be clear which years are in the difference. For that , we add two columns!
# * difference between values
# * years between that difference
# 
# sort it by date as well!

# In[ ]:


df2=df2.sort_values(by="date")
df2["year_diff"]=df2["date"].apply(lambda x : str(x.year) + '-' + str(x.year-1) )
df2["value_diff"]=df2["value"].diff()
df2.tail(10)


# **Sanity check!** 
# In years 2010 and 2009, the value was 58410 and 59118 respectively. The difference ? -708. Is that what's written in front of 2010-2009? Thankfully, Yes! 
# 
# Now lets see those peaks. Lets see where we have the highs and the lows.

# In[ ]:



temp=df2.sort_values(by="date")
plt.plot(temp["date"],temp["value_diff"])
plt.xticks(rotation=60)


# I spy four notable peaks ....
# * 1940-1949
# * 1976-1986
# * Late 90s, near the dot com bubble
# * 2001-2010 (seen the movie "The big short" anyone?)
# 
# So lets start by looking through the years -  1940-1949 .

# In[ ]:


#temp=df2[(df2["date"]>='1940') & (df2["date"]>='1950')].sort_values(by="date",ascending=False)
temp=df2[(df2["date"]>='1940') & (df2["date"]<='1950')]
plt.plot(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)


# Huge!!! any idea what it could be? While there are tons of historic events between 1940 to 1949; World War II and the independence of South East Asian states among the top picks.... None of those apply here.
# 
# At the begining of this Exploratory Data Analysis, EDA, we saw that there was no data for the years 1941-1946. It was all dots.
# So the calculation of the difference for the year 1947-1946 is actually for the year 1947-1940.
# 
# Moving on to 1976 - 1986.

# In[ ]:


temp=df2[(df2["date"]>='1976') & (df2["date"]<='1986')]
plt.plot(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)


# Ok here's something promising, we see a huge spike for the years 1979 through 1980.
# 
# Moving on to the phenomenal 90s!

# In[ ]:


temp=df2[(df2["date"]>='1990') & (df2["date"]<='2000')]
plt.bar(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)


# Spikes seen for the years 1993-1992 , 1998-1997 till the late 2000s.
# What i can remember from that era was the dot com bubble.
# Maybe people dumped a lot of money into stocks and didnt go for housing for the years 1995-1996, where we see a dip.
# 
# Lets move on to the last one, as i recall my reference to the famous movie "The big short"

# In[ ]:


temp=df2[(df2["date"]>='2004') & (df2["date"]<='2010')]
plt.bar(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)


# So subprime mortgages were a huge thing back in the 2000s, the new millenium.
# Most of the folks reading this wont be stranger to the "Rise and Fall of the Lehmann Brothers" and the "Bear-Stern stock crash".
# 
# Looking at the peaks, does it justify to include the above terms in this discussion?

# In conclusion, we can say that the dataset although has only two important attributes, it does give some neat insights if you weigh them in the context of different happenings occuring throughout these eventful (close to) 80 years!
# 
# What else can we do with this data ? How about ....
# 
# * Rise and fall stats for the presidential terms of Nixon, Reagan, Bush, Clinton .....
# * Rise and fall stats for five year periods
# * Rise and fall stats correlated with the availability of lending banks in America.
# 
# Got more? Happy to see 'em in the comments.
