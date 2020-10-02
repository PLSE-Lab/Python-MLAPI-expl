#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Crime in Chicago from 2012-2016

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


first_set = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')
# Drop rows that have 2017 as the year.
crimes = pd.DataFrame(first_set[first_set['Year'] != 2017])
crimes.head()


# # Crime Count
# - How many unique crimes are available and what is the count for each crime from 2012-2016?

# In[ ]:


# Group by Crime type and calculate count
crime_count = pd.DataFrame(crimes.groupby('Primary Type').size().sort_values(ascending=False).rename('Count').reset_index())
crime_count.head()


# In[ ]:


crime_count.shape


# In[ ]:


# Plot top 10 crimes on a barplot
crime_count[:10].plot(x='Primary Type',y='Count',kind='bar')


# # Crime Location

# In[ ]:


# Group by Crime Location and calculate count
crime_location = pd.DataFrame(crimes.groupby('Location Description').size().sort_values(ascending=False).rename('Count').reset_index())
crime_location.head()


# In[ ]:


crime_location.shape


# **Visualization**
# 
#  - There are 142 unique location descriptions, hence only the top 10 would be visualized

# In[ ]:


# Plot top 10 crime location on a barplot
crime_location[:10].plot(x='Location Description',y='Count',kind='bar')


# # Monthly Crime Activity from 2012 - 2016

# In[ ]:


import calendar
from datetime import datetime
crimes['NewDate'] =  crimes['Date'].apply(lambda x: datetime.strptime(x.split()[0],'%m/%d/%Y'))


# In[ ]:


crimes['MonthNo'] = crimes['Date'].apply(lambda x: str(x.split()[0].split('/')[0]))


# In[ ]:


monthDict = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
crimes['Month'] = crimes['MonthNo'].apply(lambda x: monthDict[x])
crimes.head()


# In[ ]:


crime_activity_plot = pd.DataFrame(crimes.groupby(['Month','Year']).size().sort_values(ascending=False).rename('Count').reset_index())
crime_activity_plot.head()


# In[ ]:


crime_activity_plot_2012_2016 = crime_activity_plot.pivot_table(values='Count',index='Month',columns='Year')


# In[ ]:


sns.heatmap(crime_activity_plot_2012_2016)


# **From the matrixplot above two observations were made:**
# 
# - 2012 and 2013 had the highest crime activity in the five-year period
# - Also, through out the five-year period, June & July had the highest crime activity
# 
# *To get a clearer picture, a clustered matrix is plotted below*

# In[ ]:


sns.clustermap(crime_activity_plot_2012_2016,cmap='coolwarm')


# **After Clustering, two observations were made:**
# 
# - The months in 2012 & 2013 with the highest crime activity are May, June, July and August
# - Also, through out the five-year period, these four months recorded the highest crime activity

# # Monthly Arrest Activity from 2012 - 2016

# In[ ]:


arrest_yearly = crimes[['Year','Arrest','Month']]
arrest_yearly_new = arrest_yearly[arrest_yearly['Arrest'] == True]
arrest_yearly_plot = pd.DataFrame(arrest_yearly_new.groupby(['Month','Year']).size().sort_values(ascending=False).rename('Count').reset_index())
arrest_yearly_plot.head()


# In[ ]:


arrest_yearly_matrix = arrest_yearly_plot.pivot_table(values='Count',index='Month',columns='Year')


# In[ ]:


sns.heatmap(arrest_yearly_matrix)


# From the matrix plot above we can see that arrests were at the highest throughout 2012-2014, and also notice a general trend which is the decrease in arrests over the years. This can be due to the fact that there has been a general decrease in crime activity during that five-year period.

# # Plotting Crime Activity and Arrest for 2012-2016
# 
# *To justify the claim above, we try to visualize the trend in arrest and crime activity side by side to get a clearer picture.*

# In[ ]:


crime_activity = pd.DataFrame(crimes.groupby('Year').size().rename('Count').reset_index())
crime_activity
arrest = crimes[['Year','Arrest']]
arrest_new = arrest[arrest['Arrest'] == True]
arrest_activity = pd.DataFrame(arrest_new.groupby('Year').size().rename('Count').reset_index())
arrest_activity


# In[ ]:


import matplotlib.ticker as ticker
x=['2012','2013','2014','2015','2016']
y=crime_activity['Count']
z=arrest_activity['Count']
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.plot(x,y,label='Crime Activity')
ax.plot(x,z,label='Arrests')
ax.set_ylabel("COUNT")
ax.set_xlabel("YEAR")
ax.set_title("Crime Activity VS Arrests from 2012 - 2016")
ax.legend()


# ## Conclusion
# - So from the line plot above, truly there has been a decrease in crime activity over the last five years which has lead to fewer arrests.
# - Also there is a wide gap between arrests and crime activity, about ~ 250,000 difference in crime activity and arrests.
# - The high rate of crime activity during May-August, might be due to the fact that it's the summer period where people are always outside and more vulnerable to attacks from perps.
# 
#  *This is my first EDA on a dataset from kaggle and any feedback, suggestions or comments would be appreciated :)*

# In[ ]:




