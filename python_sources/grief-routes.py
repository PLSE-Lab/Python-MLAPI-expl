#!/usr/bin/env python
# coding: utf-8

# # Grief Routes

# ## 1. Introduction and Preparation

# Immigration are one of the most important problems globally in recent years. It causes political issues throughout the world. Some may see them only as peoples who settled other countries and fled the war. But tey are just seeking a safe place to live in. And some of them can not make it.
# 
# Missing Migrants Project tracks deaths of migrants, including refugees and asylum-seekers, who have gone missing along mixed migration routes worldwide. The research behind this project began with the October 2013 tragedies, when at least 368 individuals died in two shipwrecks near the Italian island of Lampedusa.
# 
# This dataset from International Organization for Migration (IOM) includes events during migration between January 2014 and March 2019 and it represents minimum estimates, as many deaths during migrations are unrecorded. I am trying to show how the bad situtation is and hopefully we will get a better understanding of the circumstances of these people who have to flee for their lives

# Importing required packages

# In[ ]:


import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
        #import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# Reading and taking a look at the raw data

# In[ ]:


data = pd.read_csv("../input/missing-migrants-project/MissingMigrants-Global-2019-03-29T18-36-07.csv")


# In[ ]:


data.info()


# In[ ]:


data.head(3)


# Dropping unnecessary columns

# In[ ]:


data.drop('Reported Date', axis=1, inplace=True)
data.drop('Information Source', axis=1, inplace=True)
data.drop('URL', axis=1, inplace=True)
data.drop('UNSD Geographical Grouping', axis=1, inplace=True)


# Filling N/A values with 0 in related columns

# In[ ]:


data['Number Dead'].fillna(0, inplace=True)
data['Minimum Estimated Number of Missing'].fillna(0, inplace=True)
data['Total Dead and Missing'].fillna(0, inplace=True)
data['Number of Survivors'].fillna(0, inplace=True)
data['Number of Females'].fillna(0, inplace=True)
data['Number of Males'].fillna(0, inplace=True)
data['Number of Children'].fillna(0, inplace=True)


# Fractions of person count does not mean anything so I converting float numbers into integers

# In[ ]:


data['Number Dead'] = data['Number Dead'].astype(int)
data['Minimum Estimated Number of Missing'] = data['Minimum Estimated Number of Missing'].astype(int)
data['Total Dead and Missing'] = data['Total Dead and Missing'].astype(int)
data['Number of Survivors'] = data['Number of Survivors'].astype(int)
data['Number of Females'] = data['Number of Females'].astype(int)
data['Number of Males'] = data['Number of Males'].astype(int)
data['Number of Children'] = data['Number of Children'].astype(int)


# In[ ]:


data.head(3)


# In[ ]:


data.describe()


# ## 2. Analysis

# Now data is ready to make some analysis.

# Below bar chart shows the number of incidents and regions with at least 100 incidents

# In[ ]:


roi = data['Region of Incident'].value_counts()
roi = roi[roi > 100]
print(roi)

roi.plot(kind='barh', color = '#00b8a9')
plt.title('Regions with most incidents')
plt.ylabel('Number of Incidents')
plt.xlabel('Regions')
#plt.setp(lines, color='r', linewidth=2.0)
plt.show()


# Below line graph shows the number of deaths through years between 2014 and 2019
# 
# Because of the data for 2019 is only for 3 months, 2019 incidents are excluded to see the trend more clearly.

# In[ ]:


minus2019 = data[data['Reported Year'] != 2019]
dead = minus2019[minus2019['Number Dead'] > 0]


# In[ ]:


yearly_dead = dead.groupby('Reported Year')['Number Dead'].sum()
print(yearly_dead)


# In[ ]:


yearly_dead.plot(color = '#f67280')
plt.title('Loss of Life Through Years')
plt.ylabel('Total Loss of Life')
plt.xlabel('Year')
plt.show()


# 
# 

# Below bar chart displayes the deaths, distributed to genders, through years

# In[ ]:


#I dont know why, but this agg dropped the year column from the df or changed the level of it.
gender = dead.groupby('Reported Year').agg(
        female=('Number of Females', sum),
        male=('Number of Males', sum)
)

print(gender)


# In[ ]:


#Adding year column
year = [2014,2015,2016,2017,2018]
gender['Year'] = year


# In[ ]:


plt.bar(gender['Year'], gender['female'], color="#f3e151")
plt.bar(gender['Year'], gender['male'], bottom=gender['female'], color="#6c3376")
plt.legend(['Female','Male'])
plt.title('Total Loss of Life by gender through years')


# Cause of death

# In[ ]:


cause = data.groupby('Cause of Death')['Number Dead'].sum()
print(cause.sort_values(ascending = False))


# In[ ]:


cause = cause.nlargest(10)
cause.plot(kind='barh', color = '#4a69bb')
plt.title('Top 10 Cause of Death')
plt.ylabel('Cause of Death')
plt.xlabel('Total Loss of Life')
plt.show()


# ## 3.Conclusion

# As a conclusion, when we look at the analysis, US-Mexico Border is the deadliest place for refugees while North Africa take the second place. Total loss of life hit the peek in 2015 while most of the death were cause by drowning.

# ## 4.References

# [IOM's Missing Migrants Project website](https://missingmigrants.iom.int/)
# 
# [Kaggle Dataset](https://www.kaggle.com/snocco/missing-migrants-project)
