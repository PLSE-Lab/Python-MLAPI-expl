#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')


# **Check the columns in the dataset**

# In[ ]:


data.info()


# **Lets explore the number of Confirmed cases of Corona Virus**

# In[ ]:


country = data.groupby('Country').sum().apply(list).reset_index()
country


# In[ ]:


country.plot(kind='barh', x='Country', y='Confirmed', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)


# **As we can see from the above visualisation that the virus is concentrated in China and Mainland China**
# 
# **Lets see what is the recovery rate of the people diagnoseed with Virus**

# In[ ]:


country['%Recovered'] = ((country['Recovered'] / country['Confirmed'] ) * 100)
country['%Recovered'] = country['%Recovered'].round(2)
country


# **As we can see that the recovery rate of the patient is less than 25% worldwide, the person diagnosed with the virus has very less chance of survival.**
# 

# **Lets see how the Corona virus is spreading over time**

# In[ ]:


from datetime import datetime,date

data['date'] = pd.to_datetime(data['Last Update']).dt.date
data


# In[ ]:


date_conf_cases = data.groupby('date').sum().apply(list).reset_index()
date_conf_cases


# **As we can see we have outlier in the dataset, Date 2020-01-02,2020-03-02,2020-04-02 are outliers so we will remove it. **

# In[ ]:


date_conf_cases = date_conf_cases.iloc[1:(len(date_conf_cases)-2)]
date_conf_cases


# **Total Number of confirmed corona virus patient**

# In[ ]:


date_conf_cases['Confirmed'].sum()


# **As we can see that the number of cases of corona virus is increasing gradually over time, we will look into this aspect more**
# 
# **Lets see the population of the world and try to analyze how will it impact HUMANITY if the cure is not found in near time**
# 
# **The population of the world is around 7.8 billion and in the past 11 days the number of corona virus patient have gone upto 63987**
# 
# **Will try to apply linear regressiom to see the impact of the virus in the coming days**
# 
# **I will replace the date commencing form Jan 22 (day 1) till last day to have a numerical data and find the total number of confirmed cases over time**

# In[ ]:


i=1
tot_conf = 0
date_conf_cases['Total Confirmed Cases'] = 1
date_conf_cases['Days'] = 1
for ind in date_conf_cases.index: 
    date_conf_cases['Days'][ind] = i
    i=i+1
    date_conf_cases['Total Confirmed Cases'][ind] = date_conf_cases['Confirmed'][ind] + tot_conf
    tot_conf = date_conf_cases['Total Confirmed Cases'][ind]
date_conf_cases


# **Dropping the Unwanted Columns (date,Sno, Recovered, Deaths)**

# In[ ]:


date_conf_cases.drop(['date', 'Sno','Deaths','Recovered','Confirmed'], axis=1)


# In[ ]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[ ]:


x= date_conf_cases['Days']
y=date_conf_cases['Total Confirmed Cases']
x_matrix=x.values.reshape(-1,1)


# In[ ]:


reg.fit(x_matrix,y)


# In[ ]:


reg.score(x_matrix,y)


# In[ ]:


reg.intercept_


# In[ ]:


reg.coef_


# **The Linear model is created **
# 
# **We will try and check how the outbreak will move in coming days**
# 
# **Lets try with three data 10year(3650 days) 20years(7300 days) and 30years(10950 days)**

# In[ ]:


new_data=pd.DataFrame(data=[3650,7300,10950],columns=['Days'])
new_data


# In[ ]:


new_data_matrix=new_data.values.reshape(-1,1)
reg.predict(new_data_matrix)


# **So we have the following inferences**
# 
# **In around 10 years 21 million people will be getting affected**
# 
# **In around 20 years 43 million people will be getting affected**
# 
# **In around 30 years 64 million people will be getting affected**
# 
# **Much of the affected area will be China and India as they are densely populated**
# 
# 
