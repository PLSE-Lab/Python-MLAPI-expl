#!/usr/bin/env python
# coding: utf-8

# This is a high-level analysis of the TTC subway delay dataset from https://open.toronto.ca/dataset/ttc-subway-delay-data/. The Open Data Portal is an initiative to meet the growing demand for open data. It provides momentum to the development of technology and data literacy. Publicizing the datasets means that people will be able to download, visualize, interpret data and eventually draw insights. Browsing through the open data catalogue, I found this topic interesting while discovering the frequency of certain types of reasons that are causing subway delay. This dataset captures TTC subway delay details in terms of station, delay code, date, bound, etc. for May 2019. Drawing a statistically confident conclusion with this sample size is challenging. However, it provides some directional read on frenquency, type of delay that could affect Torontonian's everyday life. 
# 
# 
# I also came across this report https://www.cbc.ca/news/canada/toronto/ttc-subway-delays-1.4068358 written in 2017 where CBC analyzed major reasons of TTC subway delay. Not surprisingly, the findings still hold compared to May 2019 as shown in the visualization below in terms of hours of delay by the station.
# ![](https://i.cbc.ca/1.4069523.1492099175!/fileImage/httpImage/image.jpeg_gen/derivatives/original_780/ttc-chart.jpeg)
# 
# To save whoever is reading this some time, here're some takeaways:
# 1. Budget more time for your commute on Thursdays and Fridays, especially when you will be traveling southbound.
# 2. Kipling, Finch, Kennedy BD, Bloor and Sheppard stations experienced a frequent delay in May 2019. If you happen to commute from these listed stations and always found yourself being late, try heading out 10 min (totally arbitrary) earlier than your routine schedule and see if it works out.
# 3. TTC employees also cause delays
# 
# Source:https://open.toronto.ca/dataset/ttc-subway-delay-data/

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns


# To import all the necessary libraries 

# In[ ]:


ttc_data = pd.read_csv('../input/to-subway-delay/may.csv',index_col='Date',parse_dates=True)
# This step is to drop all the rows has null/ n/a data
ttc_data = ttc_data.dropna(axis=0)


# This step is to read the subway delay dataset

# In[ ]:


#Load the first 10 rows of data from the dataset
print(ttc_data.head(10))


# This dataset captures mostly categorical values such as station, bound, vehicle and delay type. Most of the following analysis will be performed based on these dimensions.

# In[ ]:


sns.set(style='darkgrid')
delay_station = sns.countplot(y='Station',data =ttc_data,order=ttc_data['Station'].value_counts().iloc[:5].index)


# **- Count of orrcurence of subway delay by station**
# The top 5 stations listed above all are major transfer stations.
# Comparing to the CBC visualization - "This is how riders lost the most time in 2016" from https://www.cbc.ca/news/canada/toronto/ttc-subway-delays-1.4068358, it's not hard to identify the overlap of stations based on frequency of delay & impact on riders.
# 

# In[ ]:


sns.set(style='darkgrid')
delay_day = sns.countplot(y='Day',data =ttc_data,order=ttc_data['Day'].value_counts().iloc[:7].index)


# **- Count of orrcurence of subway delay by day**

# In[ ]:



sns.catplot(x="Day", y="Min Delay",data=ttc_data);


# Does frequency correlate with the duration of delay? Here's a breakdown of minutes of delay by day. As shown in the visualizaiton, most of the delay is around 0-10 minutes.
# 

# In[ ]:


sns.set(style='darkgrid')
delay_bound = sns.countplot(y='Bound',data =ttc_data,order=ttc_data['Bound'].value_counts().iloc[:4].index)


# **- Count of orrcurence of subway delay by bound**

# In[ ]:


sns.set(style='darkgrid')
delay_type = sns.countplot(y='Code',data =ttc_data,order=ttc_data['Code'].value_counts().iloc[:10].index)


# There are some obvious reasons of why delay happens. As a regular subway rider myself, my personal impression is that it happens the most because of bad weather. From this dataset, the top 5 reasons that caused subway delay in May 2019 are: MUSC, MUPAA, TUSC, SUDP and MUATC. After matching the codes to a TTC document "ttc-subway-delay_codes". A more human-readible version of the reasons are:
# 1.	MUSC - Miscellaneous Speed Control
# 2.	MUPAA - Passenger Assistance Alarm Activated - No Trouble Found
# 3.	TUSC - Operator Overspeeding
# 4.	SUDP - Disorderly Patron
# 5.	MUATC - ATC Project
# Speed control due to various reasons is one of the top reasons for subway delay followed by false alarms. Riders are not the only one slowing down the subway,delay also happens when an operator violate a signal or failed to monitor the train doors. "Disorderly patrons" is the 4th most frequent reason which includes disruptive, agreesive behaviours from riders 
# 

# To further understand how subway stations are affected by different types of delay, here's a breakdown of delay type by station.

# In[ ]:


t_codes = ['SUDP','TUSC','MUPAA']
g = sns.catplot(x='Code',col='Station',col_wrap=5,data=ttc_data.loc[ttc_data['Code'].isin(t_codes)],kind='count',height=5,aspect=.9)


# Looking forward, some of the TTC development projects could relieve crowding for major stations.
# For more details:
# 
# Relief Line: https://www.ttc.ca/About_the_TTC/Projects/Future_Plans_and_Studies/Relief_Line/index.jsp
# METROLINX:http://www.metrolinx.com/en/default.aspx
# 
