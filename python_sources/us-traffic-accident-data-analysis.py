#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


# import package with helper functions 
import bq_helper


# In[ ]:


# create a helper object for this dataset
caraccidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="nhtsa_traffic_fatalities")


# In[ ]:


caraccidents.head("accident_2015")


# In[ ]:


#States with most number of accidents in 2015
query="""SELECT COUNT(Consecutive_number) AS NumberofAccidents,State_name 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY state_name
ORDER BY NumberofAccidents DESC"""

NumberofAccidentsPerState_2015=caraccidents.query_to_pandas_safe(query,21)


# In[ ]:


#States with most number of accidents in 2015
query="""SELECT COUNT(Consecutive_number) AS NumberofAccidents,State_name 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY state_name
ORDER BY NumberofAccidents DESC"""

NumberofAccidentsPerState_2016=caraccidents.query_to_pandas_safe(query,21)


# In[ ]:


NumberofAccidentsPerState_2015.head()


# In[ ]:


NumberofAccidentsPerState_2016.head()


# - Texas is leading the number of accidents reported in both 2015 and 2016.
# - The top 5 states based on the number of accidents reported remains the same for both the years

# In[ ]:





# In[ ]:


#Percentage change in accidents cases for each state
results=NumberofAccidentsPerState_2015.merge(NumberofAccidentsPerState_2016, on='State_name', how='inner')



# In[ ]:


results['Change']=results['NumberofAccidents_y']-results['NumberofAccidents_x']
#results.head()


# In[ ]:


results['PercentChange']=(results['Change']/results['NumberofAccidents_x'])*100


# In[ ]:


results.sort_values(by='PercentChange',ascending =False).head()


# In[ ]:


results.sort_values(by='PercentChange',ascending =True).head()


# - New Mexico has the highest percentage change in the number of accidents Year on Year , followed by Alaska , Hawaii, Iowa and New Hampshire.
# - Each of these states has a percentage change of more than 25%
# - Wyoming has seen a drastic fall in the number of accidents. Followed by Montana , South Dakota and Nebraska.
# - Even New York has recorded fall in the number of accident cases when compared to 2015.

# In[ ]:


#Number of crashes per month in 2015
query="""SELECT COUNT(Consecutive_number) AS NumberofAccidents,month_of_crash 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY month_of_crash
ORDER BY NumberofAccidents DESC"""

NumberofAccidentsPerMonth_2015=caraccidents.query_to_pandas_safe(query,21)
NumberofAccidentsPerMonth_2015


# In[ ]:


#Number of crashes per month in 2016
query="""SELECT COUNT(Consecutive_number) AS NumberofAccidents,month_of_crash 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY month_of_crash
ORDER BY NumberofAccidents DESC"""

NumberofAccidentsPerMonth_2016=caraccidents.query_to_pandas_safe(query,21)

NumberofAccidentsPerMonth_2016


# In[ ]:


#Number of crashes each day in 2015
query="""SELECT COUNT(Consecutive_number) AS NumberofAccidents,day_of_crash 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY day_of_crash
ORDER BY NumberofAccidents DESC"""

NumberofAccidentsPerHour_2015=caraccidents.query_to_pandas_safe(query,21)
NumberofAccidentsPerHour_2015


# In[ ]:


#Number of crashes each day in 2016
query="""SELECT COUNT(Consecutive_number) AS NumberofAccidents,day_of_crash 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY day_of_crash
ORDER BY NumberofAccidents DESC"""

NumberofAccidentsPerHour_2016=caraccidents.query_to_pandas_safe(query,21)
NumberofAccidentsPerHour_2016


# In[ ]:


#Number of crashes per hour in 2015
query="""SELECT COUNT(Consecutive_number) AS NumberofAccidents,hour_of_crash 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY hour_of_crash
ORDER BY NumberofAccidents DESC"""

NumberofAccidentsPerHour=caraccidents.query_to_pandas_safe(query,21)


# In[ ]:


NumberofAccidentsPerHour.head()


# In[ ]:


#NumberofAccidentsPerState.plot(kind="bar)
sns.barplot(x="hour_of_crash",y="NumberofAccidents",data=NumberofAccidentsPerHour,palette='coolwarm')


# In[ ]:


query="""SELECT SUM(number_of_persons_in_motor_vehicles_in_transport_mvit) AS NumberOfFatalities,state_name 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY state_name
ORDER BY  NumberOfFatalities DESC"""

NumberOfDirectFatalitiesByState=caraccidents.query_to_pandas_safe(query,21)


# In[ ]:


NumberOfDirectFatalitiesByState.head()


# In[ ]:


NumberofAccidentsPerState.head()


# In[ ]:


fig, ax =plt.subplots(figsize=(6,15),ncols=2,sharey=True)

sns.set_style("whitegrid")
sns.barplot(x="NumberOfFatalities",y="state_name",data=NumberOfDirectFatalitiesByState,dodge=False, ax=ax[0])

sns.barplot(x="NumberofAccidents",y="State_name",data=NumberofAccidentsPerState, ax=ax[1])


# In[ ]:




