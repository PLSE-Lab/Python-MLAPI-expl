#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


das = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv")
das = pd.DataFrame(das)
das.head()


# In[ ]:


das = das.rename(columns = {'SUBDIVISION':'state', 'JAN': 'jan', 'FEB': 'feb', 'MAR': 'mar', 
                       'APR': 'apr' , 'MAY': 'may', 'JUN': 'jun', 'JUL': 'jul', 'AUG': 'aug', 'SEP': 'sep', 'OCT': 'oct'
                      , 'NOV': 'nov', 'DEC': 'dec', 'ANNUAL': 'annual', 'YEAR': 'year'})
das.tail()


# ***I have changed the label of the rows as per my ease of operation***

# In[ ]:


das.state.unique()


# ***Unique values of the state are mentioned above***

# In[ ]:


das.year.unique()


# In[ ]:


das_kon = das[das.state=='KONKAN & GOA']
das_kel = das[das.state=='KERALA']
#das_kel.head()


# > ***I have created two specific data set one from KONKAN, and other for Kerla to process the data in further codes***

# In[ ]:


plt.figure(figsize = (20,5))
sns.barplot(x='year', y= 'annual', data = das_kon)
plt.xticks(rotation = 90)
plt.title('KONKAN & GOA RAINFALL DATA FROM 1901-2015')
plt.show()


# In[ ]:


plt.figure(figsize = (20,5))
sns.barplot(x='year', y= 'annual', data = das_kel)
plt.xticks(rotation = 90)
plt.title('KERALA RAINFALL DATA FROM 1901-2015')
plt.show()


# In[ ]:


plt.figure(figsize=(25,15))
sns.lineplot(x = 'year', y= 'annual', hue = 'state', data = das)


# In[ ]:


das_kon.groupby(['year'])['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'].sum().plot.line(figsize=(15,5))
plt.title('KONKAN & GOA')
das_kel.groupby(['year'])['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'].sum().plot.line(figsize=(15,5))
plt.title('KERALA')
plt.show()


# ***Above charts shows the rainfall data for two states ***

# In[ ]:


plt.figure(figsize=(15,5))
das.groupby(['state','year'])['annual'].sum().sort_values(ascending=False).plot()
plt.xticks(rotation=90)


# ***I have used groupby function and sorted the values in ascending order to create rainfall data***

# ***From the below code result and the chart above you can see that Arunachal Pradesh had highest rainfall in 1948, Coastal Karnataka had higest rainfall in 1961***

# In[ ]:


das.groupby(['state','year'])['annual'].sum().sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(15,5))
das.groupby(['year'])['annual'].sum().sort_values(ascending=False).head(12).plot(kind='bar', color = 'b')
plt.ylabel('Yearly Rainfall')
plt.title('Yearly Rainfall Data')


# **Above chart shows you that in last 115 years 1961 was the year which had highest rainfall**

# In[ ]:


plt.figure(figsize=(15,5))
das.groupby(['year'])['annual'].sum().sort_values(ascending=True).tail(12).plot(kind='bar', color = 'c')
plt.ylabel('Yearly Rainfall')
plt.title('Yearly Rainfall Data')


# ***Above chart show that the year 1970 had lowest rainfall in last 115 years***

# In[ ]:


plt.figure(figsize=(15,5))
das.groupby(['state'])['annual'].sum().sort_values(ascending=False).head(12).plot(kind='bar', color = 'g')
plt.ylabel('Total Rainfall')
plt.title('Total Rainfall Data')


# ***Above chart shows that Costal Karnataka has received highest rainfall in last 115 years***

# In[ ]:


plt.figure(figsize=(15,5))
das.groupby(['state'])['annual'].sum().sort_values(ascending=True).tail(10).plot(kind='bar', color = 'b')

plt.ylabel('Total Rainfall')
plt.title('Total Rainfall Data')


# ***Above chart shows that UTTARAKHAND has received lowest rainfall in last 115 years (This is beacuse UTTARAKHAND has been carved from UP in the year 2000, hence insufficient data )***

# In[ ]:


plt.figure(figsize=(10,5))
das[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
       'sep', 'oct', 'nov', 'dec']].mean().plot(kind= 'bar')
plt.xlabel('Months')
plt.ylabel('Avg. Rainfall')
plt.title('Avg. Monthly Rainfall Data')
plt.show()


# ***Above bar graph shows the average rainfall received from Jan-Dec in last 115 years.***
