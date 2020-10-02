#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


crime = pd.read_csv('../input/crime-classifcication/Crime1.csv',usecols=['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address'])


# In[ ]:


crime.head()


# # Data Audit

# In[ ]:


crime.info()


# In[ ]:


crime.dtypes


# In[ ]:


crime.describe()


# ### Converting 'Dates' column to proper date time format

# In[ ]:


crime['Dates'] = pd.to_datetime(crime['Dates'])


# In[ ]:


crime.dtypes


# ### Finding the count of unique values in 'Category', 'DayOfWeek', 'pdDistrict', 'Resolution' columns
# 

# In[ ]:


crime['Category'].value_counts()


# In[ ]:


crime['DayOfWeek'].value_counts()


# In[ ]:


crime['PdDistrict'].value_counts()


# In[ ]:


crime['Resolution'].value_counts()


# # Which was the most common category of crime

# In[ ]:


crime_category = crime.groupby('Category')['Category'].count().sort_values(ascending=False)


# In[ ]:


crime_category


# In[ ]:


plt.figure(figsize=(10,8))
crime_category.plot(kind='barh')
plt.xlabel('Count')
plt.title('Number of times each Crime Category took place')
plt.show()


# In[ ]:


print('We can see from the above plot that LARCENY/THEFT is the most common crime category')


# In[ ]:


crime.head()


# # On which day of the week most crimes take place

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(crime['DayOfWeek'])
plt.title('Day of the week on which most crimes take place')
plt.show()


# In[ ]:


print('On SATURDAY most crimes take place')


# # Which is the most famous district in terms of crime

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(crime['PdDistrict'])
plt.show()


# In[ ]:


print('SOUTHERN district is famous in terms of crimes')


# # What was the description in case of LARCENY/THEFT	cases

# In[ ]:


larseny_descript = crime.loc[crime['Category']=='LARCENY/THEFT','Descript'].value_counts()


# In[ ]:


larseny_descript


# # How were LARCENY/THEFT cases resolved

# In[ ]:


# grouping the data set on the basis of 'Category' and 'Resolution' columns
category_resolution = crime.groupby(['Category','Resolution'])['Category'].count()


# In[ ]:


category_resolution


# In[ ]:


# filtering out 'LARCENY/THEFT' cases
larceny_cases = category_resolution['LARCENY/THEFT']


# In[ ]:


larceny_cases


# In[ ]:


larceny_cases.plot(kind='bar')
plt.show()


# In[ ]:


print('There was no resolution for majority of LARCENY/THEFT cases')


# # On which day of the week maximum 'LARCENY/THEFT' crime cases took place

# In[ ]:


# grouping the data based on 'Category' and 'DayOfWeek' columns
category_day = crime.groupby(['Category','DayOfWeek'])['Category'].count()


# In[ ]:


category_day


# In[ ]:


# filtering out 'LARCENY/THEFT' cases
larceny_day = category_day['LARCENY/THEFT']


# In[ ]:


larceny_day


# In[ ]:


plt.figure(figsize=(10,8))
larceny_day.plot(kind='bar')
plt.ylabel('Count')
plt.show()


# # Checking if there is a specific address in 'SOUTHERN' district where LARCENY/THEFT crimes take place

# In[ ]:


larseny_address = crime.groupby(['Category','PdDistrict','Address'])['Address'].count()['LARCENY/THEFT']['SOUTHERN'].sort_values(ascending=False)


# In[ ]:


larseny_address


# In[ ]:


print('No specific address')


# In[ ]:




