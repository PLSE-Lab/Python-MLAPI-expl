#!/usr/bin/env python
# coding: utf-8

# India is a country of huge diversity, to address this diversity the constitution give rights, privilages to it's citizens. One such right is given to the the children of age group 6-14 of compulsary education under article 21A. But there are various factors that do not let these children to get education. This might be due to various reasons Eg. Poverty.
# 
# But what about the higher education which is also very important?higher education ensures a person's place in society as it imparts him/her with specialization and skills that would get him/her a job.
# 
# In the following datset of Secondary education i am attempting to understand what factors lead to higher/lower litracy rate by considering population, sex ratio, number of schools etc.
# 
# Since i'm a beginner my knowledge in this field is limited as i'm learning. I would appretiate feedback, tips, advices if any, Thank You.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


meta = pd.read_csv("/kaggle/input/education-in-india/2015_16_Statewise_Elementary_Metadata.csv")
meta


# In[ ]:


edu_sec = pd.read_csv("/kaggle/input/education-in-india/2015_16_Statewise_Secondary.csv")
edu_sec


# I dropped Telangana as there was not much information to begin with and data was also not available.

# In[ ]:


edu_sec.drop([35], inplace=True)
edu_sec


# Columns that i thought would help me understand the senario better. 

# In[ ]:


col = ['ac_year', 'statcd', 'statname', 'area_sqkm', 'tot_population',
       'urban_population', 'grwoth_rate', 'sexratio', 'sc_population',
       'st_population', 'literacy_rate', 'male_literacy_rate','female_literacy_rate', 'schools', 'villages', 'distcd']
edu_sec = pd.DataFrame(edu_sec, columns=col)
edu_sec.head()


# In[ ]:


edu_sec.columns


# In[ ]:


edu_sec = edu_sec.rename(columns={'statname': 'State', 'tot_population': 'Population'})
edu_sec.head()


# In[ ]:


edu_sec['State'] = edu_sec['State'].str.strip()


# In[ ]:


edu_sec.describe()


# I could'eve just used 
# edu_sec['Rural Population'] = (100-edu_sec['urban_population']) Here and it wouldeve worked.

# In[ ]:



edu_sec['Rural Population'] = (((100-edu_sec['urban_population'])/100)*edu_sec['Population'])
edu_sec['Rural Population'] = (edu_sec['Rural Population']*100)/edu_sec['Population']
edu_sec.head()


# Mergerd the SC and St columns for ease of une=derstanding as both the categories had only the population detaail. It would have been greater use if the litracy rate of these categories were available.

# In[ ]:


edu_sec['SC\ST Population'] = edu_sec['sc_population'] + edu_sec['st_population']
col = ['sc_population','st_population']
edu_sec.drop(col, axis=1, inplace=True)
edu_sec.head(1)


# In[ ]:


plt.figure(figsize=(10,12))
sns.barplot(data=edu_sec, x='Population', y='State',  order=edu_sec.sort_values('Population').State).set_title('Population of States', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Population', fontsize=15)
plt.ylabel('Name of States', fontsize=15)
plt.xticks(rotation=90)


# In[ ]:


edu_sec.head(2)


# In[ ]:


edu_sec.head()


# In[ ]:


plt.figure(figsize=(10,12))
sns.barplot(data=edu_sec, x='schools', y='State',  order=edu_sec.sort_values('schools').State).set_title('Number of schools in all states', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Schools', fontsize=15)
plt.ylabel('Name of States', fontsize=15)
plt.xticks(rotation=90)


# In[ ]:


edu_sec['State'].unique()


# The dataset had the records of states and UT's together so I decided to seperate then because of the difference in size of population of both. 
# 
# NOTE: Did not include J&K in UT's as iit was still a state in  2015-2016

# **UNION TERRITORIES**

# In[ ]:


uts = ['Chandigarh', 'Delhi','Daman & Diu', 'Dadra & Nagar Haveli'
      , 'Lakshadweep', 'Puducherry','Andaman & Nicobar Islands']
ut = edu_sec[(edu_sec['State'].isin(uts))].reset_index()
ut


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=ut, x='State', y='schools',order=ut.sort_values('schools').State).set_title('Number of schools in Union Territory', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Union Territory', fontsize=15)
plt.ylabel('Number of schools', fontsize=15)
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=ut, x='State', y='literacy_rate', order=ut.sort_values('literacy_rate').State).set_title("Litracy Rate of UT's", fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Union Territory', fontsize=15)
plt.ylabel('Litracy of the UT', fontsize=15)
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=ut, x='State', y='Population', order=ut.sort_values('Population').State).set_title('Population of Union Territory', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of UT', fontsize=15)
plt.ylabel('Population', fontsize=15)
plt.xticks(rotation=90)


# For UT the litracy rate of Lakshadweep is high even thouugh the population of Delhi and the number of schools is higher. Also the SC/St population is north of 85% in lakshadweep, i.e. the SC/ST Poplation is more educated in that region than others. even though their population is less compared mto say Delhi the litraacy rate is suerly higher.

# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=ut, x='State', y='SC\ST Population',  order=ut.sort_values('SC\ST Population').State).set_title('Population of States', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of States', fontsize=15)
plt.ylabel('SC\ST Population', fontsize=15)
plt.xticks(rotation=90)


# ******STATES**

# In[ ]:


state = ['Jammu And Kashmir', 'Himachal Pradesh', 'Punjab',
       'Uttarakhand', 'Haryana',
       'Rajasthan', 'Uttar Pradesh', 'Bihar',
       'Sikkim', 'Arunachal Pradesh', 'Nagaland', 'Manipur',
       'Mizoram', 'Tripura',
       'Meghalaya', 'Assam',
       'West Bengal', 'Jharkhand', 'Odisha',
       'Chhattisgarh', 'MADHYA PRADESH', 'Gujarat',
        'Maharashtra', 'Andhra Pradesh', 'Karnataka',
       'Goa', 'Kerala', 'Tamil Nadu', 'Telangana']
states = edu_sec[(edu_sec['State'].isin(state))].reset_index()
states 


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=states, x='State', y='schools', order=states.sort_values('schools').State).set_title('Number of schools', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of States', fontsize=15)
plt.ylabel('Number of schools', fontsize=15)
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=states, x='State', y='literacy_rate', order=states.sort_values('literacy_rate').State).set_title('Litracy rate  of States', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of States', fontsize=15)
plt.ylabel('Litracy Rate of states', fontsize=15)
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=states, x='State', y='Population',  order=states.sort_values('Population').State).set_title('Population of States', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of States', fontsize=15)
plt.ylabel('Population', fontsize=15)
plt.xticks(rotation=90)


# it is seen that the litracy rate of the north_eastern states is high even though the population and the number of schools id lower than other state.

# In[ ]:


edu_sec.head()


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=states, x='State', y='SC\ST Population',  order=states.sort_values('SC\ST Population').State).set_title('SC/ST Population of States', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of States', fontsize=15)
plt.ylabel('SC\ST Population', fontsize=15)
plt.xticks(rotation=90)


# Besides Mizoram, Tripura, Sikkim the rest of the N-E states have litracy rate lower than 80%. even though the states mentioned have a litracy rate of >80% the SC\ST Population of mizoram is little over 85%. this means the SC\ST populatio of mizoram is most literate than the rest of the countries SC\ST population. Here it can be noted that even though kerela has heighest litracy rate the same can not be said for its SC\ST population because of the low SC\ST population.

# In[ ]:


edu_sec.head()


# In[ ]:


col  = ['ac_year', 'statcd', 'State', 'area_sqkm', 'Population',
       'urban_population', 'Rural Population', 'SC\ST Population', 'grwoth_rate', 'sexratio', 'literacy_rate',
       'male_literacy_rate', 'female_literacy_rate', 'schools', 'villages',
       'distcd']

edu_sec = edu_sec.reindex(columns=col)
edu_sec.head()


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=states, x='State', y='Rural Population',  order=states.sort_values('Rural Population').State).set_title('Rural Population of States', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of States', fontsize=15)
plt.ylabel('Rural Population', fontsize=15)
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(data=states, x='State', y='sexratio',  order=states.sort_values('sexratio').State).set_title('Sex Ratios of States', fontsize=20)
sns.set_style('darkgrid')
plt.xlabel('Name of States', fontsize=15)
plt.ylabel('Sex Ratio', fontsize=15)
plt.xticks(rotation=90)


# It can be senn here that the litracy rate of states with high sex ratio is better.Yet there is a peculiar state Chhattisgarh which is among the heighest in sex ratio but thee litracy rate is low. this could be because of the naxal problem in that region.

# Thoughts: As we know India a huge country with vast diversity to address this diversity is intself an herculean task. Yet when we see education and litracy of India we observe that the stated which are more devloped give more opportunity to the educated and offers a better life. These staes are those on the wester side of India (Maharashtra, Gujrat, Karnataka, Kerela). The sates on the northern side don't see much industrial devlopment adn are mostly agricultural region as rivers like Ganga, Yamuna flows through them and give better oppoutunity at agricultural activities so education is not so much important. For the states on the Eastern side like Chattisgarh, Odisha are plagues with the Naxal problem which does not allow industrial activity. The states of North-Eastern region seem to do well.
