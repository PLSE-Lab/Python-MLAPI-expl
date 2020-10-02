#!/usr/bin/env python
# coding: utf-8

# TASK: Explore whether the no. of cases, deaths of malaria increases every year?

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
print('Setup Completed')


# LOAD AND READ DATA
# 

# In[ ]:


# Reported csv file contains, reported number of cases across the world
file_path = '../input/malaria-dataset/reported_numbers.csv'
reported_data =pd.read_csv(file_path)
reported_data.head()


# In[ ]:


# ASSESSING DATA
reported_data.info()


# In[ ]:


reported_data.columns


# from the above info,the reported_data contains column with missing value on 'No. of cases' and 'No. of deaths' columns let fix that.

# In[ ]:


# filling in missing data with mean() of that column
reported_data.fillna(value={"No. of cases":reported_data["No. of cases"].mean(), "No. of deaths":reported_data["No. of deaths"].mean()}, inplace=True)
reported_data.Year.astype(int)
# check for missing values
print('\n Total number of null values:\n',reported_data.isnull().sum())


# In[ ]:


reported_data.head(-10)


# EDA

# In[ ]:


# Q. Total number of malaria cases reported worldwide 2000 - 2017

print('\n Total Number of Cases reported worldwide:\n ',reported_data['No. of cases'].sum().astype(int))


# In[ ]:


# Q. Total number of malaria death reported worldwide from 2000 - 2017

print('\n Total Number of deaths reported worldwide:\n ',reported_data['No. of deaths'].sum().astype(int))

The Dataset used for this analysis contains reported number of Malaria cases and Malaria death from different region around the world, from the analysis above, this dataset contains a total of 757635608 reported cases and a total of  2506620 reported deaths worldwide.
# In[ ]:


# increase of malaria cases

sns.set(rc={'figure.figsize':(14,4)})

sns.set(style='whitegrid')
ax = sns.lineplot(y=reported_data['No. of cases'],x=reported_data['Year'], palette='Blues_d', data=reported_data, label='Cases')
ax.set(ylim=(0,None))
plt.title("Reported Number of Malaria Cases Worldwide (2000 - 2017)", loc='left')

The line graph above shows the exponential increase in number of Malaria cases from year 2000 to 2017.

The Plot shows a steady increase from 2000 to mid 2010 and the numbers of cases took a peak from mid 2010 to 2017, with it heighest peak in 2017 with more than 1000000 reported cases worldwide. 
# In[ ]:


sns.set(rc={'figure.figsize':(14,4)})
sns.set(style='whitegrid')
ax = sns.barplot(y=reported_data['No. of cases'],x=reported_data['WHO Region'], palette='Blues_d', data=reported_data)
plt.title("Reported Number of Malaria Cases Worldwide (2000 - 2017) Region Most affected", loc='left')

With increase in Maleria cases, Africa is the most affected region with more than 800000 reported cases, followed by South-East-Asia. 
# In[ ]:


# increase of malaria death
# pd.to_numeric('year')
sns.set(rc={'figure.figsize':(14,4)})
sns.set(style='whitegrid')
ax = sns.lineplot(y=reported_data['No. of deaths'],x=reported_data['Year'], palette='Blues_d', data=reported_data, label='Deaths')
plt.title("Reported Number of Malaria Deaths Worldwide (2000 - 2017)", loc='left')


# In[ ]:


# Q. which region has most number of deaths
sns.set(rc={'figure.figsize':(14,4)})
sns.set(style='whitegrid')
ax = sns.barplot(y=reported_data['No. of deaths'],x=reported_data['WHO Region'], palette='Blues_d', data=reported_data,)
plt.title("Reported Number of Malaria Deaths Region Most Affected (2000 - 2017)", loc='left')



# Reported Number of Malaria Deaths worldwide.
# 
# As the number of Malaria cases increased exponentially within thesame period, the number of reported deaths decreased to a minimum in 2007 and 2016 - 2017 and was at it peak in 2002.
# 
# With more number of cases, Africa remains the region with the heighest number of deaths but that was different with South-East-Asia(the region reported low number of deaths).
In Conclusion:

The analysis shows a steady increase in number of Malaria Cases and a stead decrease in number of reported Deaths. As expected in the case of Africa, more number of cases resulted to more number of death, what went wrong?, why was the case differnt for South-East-Asia?, this dataset doesn't contain much data for such analysis, pardon me for that but, it's a question i wish to answer as well.

thank you.