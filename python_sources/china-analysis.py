#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df_2019=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')


# In[ ]:


df_2019.columns


# In[ ]:


df_2019.describe()


# In[ ]:


df_2019['Time from Start to Finish (seconds)'].describe()


#  we got 19718 respondents,it's a lot.and the time is varied.and the top one got 450s to finish.

# In[ ]:


df_2019['Q1'].describe()


# we got 12 unique age ranges.and the most frequent age is 25-29.semmed like data scientist looks so young and very talented.

# In[ ]:


df_data=df_2019.loc[:,["Q1",'Q2','Q3','Q4','Q5','Q10','Q11']]


# In[ ]:


df_data.columns=['Age','Gender','Country','Education','Title','Salary','Spending']


# In[ ]:


df_data


# In[ ]:


df_data['Country'].unique()


# In[ ]:


df_data['Country']=df_data['Country'].str.replace('United kingdom of Great Britain and Northern Ireland','UK',regex=False)
df_data['Country']=df_data['Country'].str.replace('United States of America','US',regex=False)
df_data['Country']=df_data['Country'].str.replace('Iran, Islamic Republic of...','Iran',regex=False)


# In[ ]:


df_data['Country']=df_data['Country'].str.replace('Republic of Korea','South Korea',regex=False)
df_data['Country']=df_data['Country'].str.replace('Viet Nam','Vietam',regex=False)


# In[ ]:


df_data['Country']=df_data['Country'].str.replace('United Kingdom of Great Britain and Northern Ireland','UK',regex=False)


# In[ ]:


plt.plot(df_data['Country'].value_counts()[:10])


# India has the most kaggler in the world.that's really so impressive.the 2nd is US.and in japan and Russia,china,only less than 1000 respondent.seemed like these kagglers are not so activated in taking the survey.

# In[ ]:


plt.plot(df_data['Gender'].value_counts()[:2])


# In[ ]:


looks like men are far more than women.and we should make more women code!


# In[ ]:


plt.plot(df_data['Education'].value_counts()[:3])


# Most DS has a master degree. and some got Bachelor degree and little got Doctoral Degree.

# In[ ]:


df_data_china=df_data['Country']=='China'


# In[ ]:


df_data_china.value_counts()


# 574 DS get the survey

# In[ ]:


data_china=df_data[df_data['Country']=='China']


# In[ ]:


plt.plot(data_china['Gender'].value_counts())


# In[ ]:


plt.plot(data_china['Education'].value_counts()[:3])


# looks like students are made part of the kagglers in china.
# 

# In[ ]:


data_china['Title'].value_counts()[:5].plot()


# In[ ]:


data_china['Salary'].value_counts()[:3].plot()


# In[ ]:


data_china['Spending'].value_counts()[:3].plot()


# the results are kind of amazing...

# thansk for wacthing!****
