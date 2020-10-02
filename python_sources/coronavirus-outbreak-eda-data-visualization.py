#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <h3> Understanding the data

# In[ ]:


patient = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')


# In[ ]:


patient.info()


# In[ ]:


patient.isnull().sum()


# As we can see , our dataset is loaded with null values. Now let's have a peek and see what we can understand from our dataset.
# 

# In[ ]:


patient.head(10)


# In[ ]:


patient.tail(10)


# <H3>Data Visualization</H3>

# In[ ]:


sns.set(rc={'figure.figsize':(15,15)})
sns.countplot(
    y=patient['region'],

).set_title('Regions affected Overall')


# In[ ]:


reason = [x for x in patient['infection_reason'].unique()]
size = [len((patient['infection_reason'].loc[patient['infection_reason']==reason])) for reason in reason]


# Another form of visualization to see the reason how people got infected.

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(size,labels=reason, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend()
ax1.set_title('Reasons CoronaVirus\n\n')
plt.show()


# We see that majority of them got infected by ,  
# * Contact with a patient
# * Visit to Daegu
# * Visit to Wuhan

# Let's see the number of people who had contact with a cv+ patient's ,and see how are they doing.
# 

# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=patient['state'].loc[
    (patient['infection_reason']=='contact with patient')
])


# We can see that : - 
# * Majority of them are in an isolated state
# * About 12 of them were released
# * Few of them are deceased.

# Let's go more deeper and understand the spread based on 
# * Each country
# * Sex
# * Confirmed Cases

# In[ ]:


patient['country'].unique()


# <h3>Understanding the spread of the virus in China<h3>

# In[ ]:


patient['age'] = 2020-patient['birth_year']


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=patient['sex'].loc[(patient['country']=="China")]).set_title('Affected population , By gender')


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=patient['state'].loc[(patient['country']=="China") &
                                    (patient['sex']=="female")]).set_title('Female state in china')


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=patient['state'].loc[(patient['country']=="China") &
                                    (patient['sex']=="male")]).set_title('Male state in china')


# In[ ]:


sns.distplot(patient['birth_year'].loc[
    (patient['country']=="China") &
    (patient['sex']=="female")
    
]).set_title("Distribution plot for year , Females in China")


# In[ ]:


sns.distplot(patient['birth_year'].loc[
    (patient['country']=="China") &
    (patient['sex']=="male")
    
]).set_title('Distribution plot for birth year , Males in China')


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.distplot(patient['age'].loc[
    (patient['country']=="China") &
    (patient['sex']=="male")
    
]).set_title('Distribution plot for age , Males in China')


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.distplot(patient['age'].loc[
    (patient['country']=="China") &
    (patient['sex']=="female")
    
]).set_title('Distribution plot for age , Females in China')


# In[ ]:


sns.countplot(
    patient['region'].loc[
        (patient['country']=="China") 
    ]
).set_title('Regions in china where the patient got affected')


# In[ ]:


sns.set(rc={'figure.figsize':(10,10)})
sns.countplot(
    y = patient['confirmed_date'].loc[
        (patient['country']=="China")
    ]

).set_title('Confirmed dates in China')


# We see that there are very few cases in China.
# 
# Now let's take a look at Korea and see what we can find out.

# <h3>Understanding the spread of the virus in Korea<h3>

# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=patient['sex'].loc[(patient['country']=="Korea")]).set_title('Affected population , By gender in Korea')


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=patient['state'].loc[(patient['country']=="Korea") &
                                    (patient['sex']=="female")]).set_title('Female state in Korea')


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=patient['state'].loc[(patient['country']=="Korea") &
                                    (patient['sex']=="male")]).set_title('Male state in Korea')


# <h3>Distribution plot for birth year in Korea

# In[ ]:


sns.distplot(patient['age'].loc[
    (patient['country']=="Korea") &
    (patient['sex']=="female")
    
]).set_title("Distribution plot for age , Females in Korea")


# In[ ]:


sns.distplot(patient['birth_year'].loc[
    (patient['country']=="Korea") &
    (patient['sex']=="female")
    
]).set_title("Distribution plot for year , Females in Korea")


# In[ ]:


sns.distplot(patient['age'].loc[
    (patient['country']=="Korea") &
    (patient['sex']=="male")
    
]).set_title("Distribution plot for age , Males in Korea")


# In[ ]:


sns.distplot(patient['birth_year'].loc[
    (patient['country']=="Korea") &
    (patient['sex']=="male")
    
]).set_title('Distribution plot for birth year , Males in Korea')


# The average age of both male and female who got affected is in the range of 30-60

# <h3>Regions affected in Korea

# In[ ]:


sns.set(rc={'figure.figsize':(15,15)})
sns.countplot(
    y=patient['region'].loc[
        (patient['country']=="Korea")],

).set_title('Regions affected in Korea')


# In[ ]:


region_korea = [x for x in patient['region'].loc[patient['country']=="Korea"].unique()]
size_region_korea = [len(patient['region'].loc[(patient['region']==region)])
                     for region in region_korea]


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(size_region_korea,labels=region_korea, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend()
ax1.set_title('Regions in Korea\n\n')
plt.show()


# <h3>Reasons for infection in Korea

# In[ ]:


sns.set(rc={'figure.figsize':(15,15)})
sns.countplot(
    y=patient['infection_reason'].loc[
        (patient['country']=="Korea")],

).set_title('Infection reason in Korea')


# In[ ]:


infection_reason = [x for x in patient['infection_reason'].loc[patient['country']=="Korea"].unique()]
size_infection_korea = [len(patient['infection_reason'].loc[(patient['infection_reason']==infection_reason)])
                     for infection_reason in infection_reason]


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(size_infection_korea,labels=infection_reason, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend()
ax1.set_title('Reason for getting infection in Korea\n\n')
plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(15,15)})
sns.countplot(
    y=patient['confirmed_date'].loc[
        (patient['country']=="Korea")],

).set_title('Confirmed dates in Korea')


# In Korea , the confirmed cases is much higher than both of the countries.
# 
# On 2020-03-01 , the confirmed cases was the highest about 1000+ and has reduced after.
# 

# <H3>Summary</H3>
# 
# We see that the most common form of getting infected is mainly by contact.
# 
# So guys please take the necessary precautions , as this easily spreadable , and take care of yourself! 
# 
# Thanks!:)

# In[ ]:




