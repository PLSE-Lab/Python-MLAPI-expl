#!/usr/bin/env python
# coding: utf-8

# # Foreign students learning English in Malta

# ### Goals
# 
# 1. Screen Malta for national statistical office which publishes annually statistical data on English language students undertaking English language courses in Malta.
# 2. Use the data published in 2019 and prepare a short analysis.
# 3. Focus on analysis of data on Italy (as a country sending the highest number of English language students).

# ### Datasource
# 
# Source: https://nso.gov.mt/en/News_Releases/View_by_Unit/Unit_C4/Education_and_Information_Society_Statistics/Pages/Teaching-English-as-a-Foreign-Language.aspx

# # Analysis

# ## Imports and connecting to XLS file

# In[ ]:


import pandas as pd # data processing
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Number of foreign students by country for 2017-2018

# In[ ]:


df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 1_2')

df1 = df_raw.copy()[3:23]
df1.columns=['Country','Males 2017', 'Females 2017', 'Total 2017','Males 2018', 'Females 2018', 'Total 2018']
df1 = df1.set_index('Country')

df1[['Total 2017','Total 2018']][:10].plot(kind='bar',figsize=(15,10),rot=0)
plt.xlabel('Country')
plt.ylabel('Number of foreign students')
plt.title('Number of foreign students by country in 2017-2018')
plt.show()


# *As we can see from the figure above, the number of foreign students in 2017-2018 slightly fluctuates, but on average stays on the same level. Since we are interested in Italy, as the leading country by the number of foreign students, we can observe a 10% decline. *

# ## Gender distribution of foreign students from Italy in 2017-2018

# In[ ]:


fig, (ax, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15,10))
df1.loc['Italy'][['Females 2017', 'Females 2018']].plot(rot=0, kind='bar', x='LABEL', ax=ax, title=('Female students from Italy'))
df1.loc['Italy'][['Males 2017', 'Males 2018']].plot(rot=0,kind='bar', x='LABEL',  legend=False, ax=ax2, title='Male students from Italy', color='grey')
plt.show()


# The figure shows, the decrease in numbers was approximately the same both for male and female students. Also, there is a slight gender gap around 10% with female students prevailing.

# ## Foreign students distribution by age and country

# In[ ]:


df2 = df_raw.copy()[32:]
df2.columns = df2.iloc[0]
df2.rename(columns={ df2.columns[0]: "Age" }, inplace=True)
df2 = df2[1:7].set_index('Age')
df2.columns.name='Country'
df2.plot(kind='bar', figsize=(15,10),rot=0)

plt.xlabel('Age group')
plt.ylabel('Number of foreign students')
plt.title('Number of foreign students by country and age in 2018')
plt.show()


# From the figure we can conclude, the majority of students from Italy come to Malta at the age of 17 and below (around 74 % of all Italian students). This fact is quite interesting since this tendency cannot be observed for age grouping for other countries, where such a significant age gap is absent. Generally speaking, we can see only one common feature, the biggest age group from all countries is students of 15 years old and under.

# ## Foreign students by type of course followed and citizenship

# In[ ]:


df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 5')

## data preparation

df3 = df_raw.copy()[1:]
df3.columns = df3.iloc[0]
df3 = df3.set_index('Citizenship')
df3.columns.name='Type of Course'
df3=df3.iloc[1:21,[1,2,3]]
df3.rename(columns={'English specific purposes2':'English specific purposes','Other3':'Other'}, index={'Other countries4':'Other countries'},inplace=True)

## plotting

df3[:10].plot(kind='bar',figsize=(15,10),rot=0)
plt.xlabel('Country')
plt.ylabel('Number of foreign students')
plt.title('Foreign students by type of course followed and citizenship')
plt.show()


# The figure demonstrates the fact that most students come to Malta for learning Intensive English classes purposes, except Italian students. One possible explanation could be connected with the inaccurate definition of course types, since *'Other'* does not provide much valuable information about course specification as well as *'Special English'*.

# ## English students number by month

# In[ ]:


df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 6')

## data preparation

df4 = df_raw.copy()
df4.columns = df4.iloc[2]
df4.rename(columns={ df4.columns[0]: "Month" }, inplace=True)
df4 = df4.set_index('Month')
df4.columns.name='Gender'
df4=df4.iloc[3:15,[0,1]]

df4.plot.bar(figsize=(15,10),stacked=True,rot=0)
plt.xlabel('Month')
plt.ylabel('Number of foreign students')
plt.title('Number of English students per month and gender')
plt.show()


# The hottest time for students to come to Malta for learning English is July, which is almost two times more popular month than following June and August. The least popular months are December, January and February since the most number of English students combine their visits for both educational and recreational purposes during summertime.

# ## Total number of foreign student weeks by country and age group

# In[ ]:


df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 9')

## data preparation

df5 = df_raw.copy()
df5.columns = df5.iloc[1]
df5 = df5.set_index('Citizenship')
df5=df5.iloc[2:22,[0,1,2,3,4,5]]
df5.columns.name='Age'
df5.rename(index={'Other countries2':'Other countries'},inplace=True)

## plotting

df5[:5].plot(kind='bar',figsize=(15,10),rot=0)
plt.xlabel('Country')
plt.ylabel('Total number of foreign student weeks')
plt.title('Total number of foreign student weeks by country and age group')
plt.show()


# The figure above suggests, that there is a correlation between a total number of weeks a student spends learning English in Malta and his/her age. Generally speaking, the majority of students coming to Malta are 25 years old and younger, there are about 75%, on average. If we concentrate on Italy, this tendency is the most perceptible, as we saw before the number of students coming to Malta by country and their age grouping.

# ## Number of staff members by type of employment and gender

# In[ ]:


df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 12')

## data preparation

df6 = df_raw.copy()
df6.columns = df6.iloc[2]
df6.rename(columns={ df6.columns[0]: "Type of employment" }, inplace=True)
df6 = df6.set_index('Type of employment')
df6.columns.name='Gender'
df6=df6.iloc[3:6,[6,7]]



df6.plot(kind='bar',rot=0,figsize=(15,8))
plt.xlabel('Type of employment')
plt.ylabel('Number of staff members')
plt.title('Number of staff members by type of employment and gender')
plt.show()


# Speaking of the figure above, we must notice a significant gender gap in teaching staff, part-time teaching jobs are dominated by female teachers with around 70% over male teachers.
