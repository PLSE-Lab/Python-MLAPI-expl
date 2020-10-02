#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# In[ ]:


ad = pd.read_csv("/kaggle/input/uncover/UNCOVER/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv")  


# In[ ]:


ad.head()


# In[ ]:


ad.describe()


# In[ ]:


ad.dtypes


# In[ ]:


ad['tempdate'] = pd.to_datetime(ad['date']) # Creating a new date column with dtype datetime


# In[ ]:


ad.dtypes


# In[ ]:


ad.head()


# In[ ]:


ad['tempdate'].max() # Last date till which data is present


# In[ ]:


ad['tempdate'].min() # First date from which data is present


# In[ ]:


ad.columns


# In[ ]:


ad.head()


# In[ ]:


ad.fillna(0, inplace=True) # Replacing all the NaN with 0


# In[ ]:


#ad[(ad.hospitalizedcurrently > 0) & (ad.state == 'NY')]
#ad[(ad.date > '2020-03-15') & (ad.date < '2020-03-25') & (ad.state == 'NY')]
ad[ad.state == 'NY'].head() # Checking sample data from NY state


# In[ ]:


gr = ad.groupby('tempdate')


# In[ ]:


gr.first()


# In[ ]:


gr.get_group('2020-04-28')# Printing the sample grouped data of a date i.e 2020-04-28 


# In[ ]:


us_datewise = gr.sum()


# In[ ]:





# In[ ]:


us_datewise.tail()


# In[ ]:


import matplotlib.pyplot as plt


# We grouped by date and found the Sum of all attributes accross the states in USA
# 
# Objective-1: To plot the daily deaths in USA so as to understand from when the deaths started in USA and how did the trend go

# In[ ]:


fig = plt.figure(figsize = (20,10))
chart = sns.barplot(x= us_datewise.index, y="deathincrease" , 
            data = us_datewise)
chart.set_xticklabels(us_datewise.index.date,rotation=90)
chart.set_xlabel("Date",fontsize = 50)
chart.set_ylabel("No of Deaths",fontsize = 50)
chart.set_title("Daily Deaths in USA", fontsize = 50)


# Observation for Objective-1:
# Deaths started in USA from 2020-03-01 and the trend was increaing rapidly

# Objective-2: To Groupby statewise and find the mean of daily new cases so as to analyze which state was the worst affected

# In[ ]:


gr1 = ad.groupby('state')
us_statewise = gr1.mean()
us_statewise.head()


# In[ ]:


fig = plt.figure(figsize = (20,10))
chart1 = sns.barplot(x= us_statewise.index, y="positiveincrease" , data = us_statewise)
chart1.set_xticklabels(us_statewise.index,rotation=45,fontsize = 15)
chart1.set_xlabel("State",fontsize = 50)
chart1.set_ylabel("Average Daily New cases",fontsize = 30)
chart1.set_title("Average Daily New cases in different states of USA", fontsize = 40)


# Observation for Objective-2: From statewise mean data, it is understood that NY state was the worst affected follwed by NJ

# Objective-3: From the USA country-wise data to analyze the recovered vs deaths so as to find the health index of the country

# In[ ]:


us_datewise.columns


# In[ ]:


us_datewise['%recovery_outof_outcome'] = us_datewise.recovered/(us_datewise.recovered + us_datewise.death)
us_datewise['%death_outof_outcome'] = us_datewise.death/(us_datewise.recovered + us_datewise.death)


# In[ ]:


us_datewise['%recovery_outof_outcome'].fillna(1, inplace=True) # Replacing all the NaN with 1 in %recovery_outof_outcome
us_datewise['%death_outof_outcome'].fillna(0, inplace=True) # Replacing all the NaN with 0 in %death_outof_outcome
us_datewise.head()


# In[ ]:


us_datewise['%recovery_outof_outcome'].plot()
us_datewise['%death_outof_outcome'].plot()


# Observation from Objective-3: It is noted that from late February deaths started recording but no recoveries were recorded till late March, however from March ending, recoveries started and by April the health index improved and towards the end of April it further improved.

# In[ ]:




