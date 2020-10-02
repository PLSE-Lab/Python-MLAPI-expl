#!/usr/bin/env python
# coding: utf-8

# **This is review of Flights delayed data of 2008 by using Pandas.At the end, few questions will be answered through using functions, grouping and summary tables.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


flights_df=pd.read_csv('../input/2008.csv')


# In[ ]:


flights_df.head()


# In[ ]:


flights_df.columns


# In[ ]:


flights_df['DepHour'] = flights_df['DepTime'] // 100
flights_df['DepHour'].replace(to_replace=24, value=0, inplace=True)


# In[ ]:


flights_df.head()


# **1. How many unique carriers are there in our dataset?**

# In[ ]:


flights_df['UniqueCarrier'].nunique()


# **2.We have both cancelled and completed flights in the datset. Check if there are more completed or cancelled flights. What is the difference? **

# In[ ]:


list1=flights_df['Cancelled'].value_counts()
Answer= list1[0]-list1[1]
Answer


# 
# **3. Find a flight with the longest departure delays and a flight with the longest arrival delay. Do they have the same destination airport, and if yes, what is its code?**

# In[ ]:


flights_df.groupby(by=['ArrDelay','DepDelay'])['Dest'].max()
#flights_df.loc[flights_df['ArrDelay'].max() and flights_df['DepDelay'].max()]


# **4. Find the carrier that has the greatest number of cancelled flights.**

# In[ ]:


Cancelled_flights_of_each_carrier=flights_df.groupby(['UniqueCarrier','Cancelled'])
Cancelled_flights_of_each_carrier.size()


# **5. Let's examine departure time and consider distribution by hour (column DepHour that we've created earlier). Which hour has the highest percentage of flights?**

# In[ ]:


pd.crosstab(flights_df['DepHour'],flights_df['UniqueCarrier'],margins=True)


# **6. OK, now let's examine cancelled flight distribution by time. Which hour has the least percentage of cancelled flights?**

# In[ ]:


Least_percentage_of_cancelled_flights=flights_df.groupby(['Cancelled','DepHour'])
Least_percentage_of_cancelled_flights.size()
#pd.pivot_table(flights_df,index=['DepHour','Cancelled'],columns='UniqueCarrier')
#pd.crosstab(Carrier_hour,Cancelled_flights_of_each_carrier,margins=True)


# **9.Since we know the departure hour, it might be interesting to examine the average delay for corresponding hour. Are there any cases, when the planes on average departed earlier than they should have done? And if yes, at what departure hours did it happen?**

# In[ ]:


flights_df[(flights_df['DepDelay']< 0) & (flights_df['Cancelled'] == 0)]['DepHour'].value_counts()


# **10. Considering only the completed flights by the carrier, that you have found in Question 4, find the distribution of these flights by hour. At what time does the greatest number of its planes depart?**

# In[ ]:


flights_df[flights_df['Cancelled']==0]['DepHour'].value_counts()
#Cancelled_flights_of_each_carrier=flights_df.groupby(['UniqueCarrier','Cancelled'])
#Cancelled_flights_of_each_carrier.size()


# **11.Find top-10 carriers in terms of the number of completed flights (UniqueCarrier column)?**

# In[ ]:


#Top_10 = flights_df.groupby(['UniqueCarrier','Cancelled'])
#Top_10.size()
flights_df[flights_df['Cancelled']==0]['UniqueCarrier'].value_counts()


# **12. Plot distributions of flight cancellation reasons (CancellationCode).
# 
# What is the most frequent reason for flight cancellation? (Use this link to translate codes into reasons)**

# In[ ]:


flights_df['CancellationCode'].value_counts()


# **13.Which route is the most frequent, in terms of the number of flights?
# 
# (Take a look at 'Origin' and 'Dest' features. Consider A->B and B->A directions as different routes)**

# In[ ]:


flights_df['route']=flights_df['Origin'] + flights_df['Dest']


# In[ ]:


flights_df['route'].value_counts()


# **14.Find top-5 delayed routes (count how many times they were delayed on departure). From all flights on these 5 routes, count all flights with weather conditions contributing to a delay.**

# In[ ]:


#List1={}
#List1=flights_df[ (flights_df['DepDelay'] > 0)]['route'].value_counts()
#List2=List1[:5].keys()
DataFrame_crosstab=pd.crosstab(index=[flights_df['DepDelay'] > 0,flights_df['CancellationCode']],columns=(flights_df['route']),margins=True)
Table=DataFrame_crosstab.iloc[4:8,:]


# In[ ]:


Table2=Table.sort_values(by=['All'][:],axis=0,ascending=True).head().T
Table2.sort_values(by=['All'],ascending=False)


# **15. Examine the hourly distribution of departure times. Choose all correct statements**

# In[ ]:


flights_df['DepHour'].plot.hist()


# **Which month has the greatest number of cancellations due to Carrier?**

# In[ ]:


flights_df[flights_df['CancellationCode'] == 'A']['Month'].value_counts()


# ** Identify the carrier with the greatest number of cancellations due to carrier in the corresponding month from the previous question**

# In[ ]:


flights_df[(flights_df['CancellationCode'] == 'A') & (flights_df['Month'] == 4)]['UniqueCarrier'].value_counts()


# In[ ]:




