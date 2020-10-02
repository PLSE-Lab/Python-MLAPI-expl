#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px # data visualization
import matplotlib.pyplot as plt # visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id = "sec-1"> </a>
# ## Reading the Datasets

# In[ ]:


path = '/kaggle/input/coronavirusdataset/'


# In[ ]:


case_data = pd.read_csv(path+"Case.csv")
case_data.head()


# In[ ]:


patient_info = pd.read_csv(path+"PatientInfo.csv")
patient_info.head()


# <a id = "sec-two"> </a>
# ## EDA of Cases
# 1. [Case Data Information: What do we know?](#sec-2aa)
# 2. [Which Pronvice has most confirmed cases?](#sec-2a)
# 3. [Which Pronvice has least confirmed cases?](#sec-2b)
# 4. [What infection case has most number of confirmed cases?](#sec-2c)
# 5. [What percentage are Group infections?](#sec-2d)

# <a id = "sec-2aa"> </a>
# ### Case Data Information: What do we know?

# In[ ]:


case_data.info()


# <a id = "sec-2a"> </a>
# ### Which Pronvice has most confirmed cases? ----------> Daegu

# In[ ]:


provinces = case_data.groupby('province')['confirmed'].sum().reset_index()
most_provinces = provinces.sort_values('confirmed', ascending = False)[0:5]

fig = px.bar(most_provinces,x="province", y="confirmed", barmode='group')
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.5, width = 0.3)
fig.update_layout(title_text='Provinces with most confirmed cases')
fig.show()


# <a id = "sec-2b"> </a>
# ### Which Pronvice has least confirmed cases? ----------> Jeju-Do

# In[ ]:


least_provinces = case_data.groupby('province')['confirmed'].sum().reset_index()
least_provinces = least_provinces.sort_values('confirmed')[0:5]

fig = px.bar(least_provinces,x="province", y="confirmed", barmode='group')
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.5, width = 0.3)
fig.update_layout(title_text='Provinces with least confirmed cases')
fig.show()


# <a id = 'sec-2c'> </a>
# ### What infection case has most number of confirmed cases? ----------> Shincheonji Church

# In[ ]:


infection_cases = case_data.groupby('infection_case')['confirmed'].sum().reset_index()
infection_cases = infection_cases.sort_values('confirmed', ascending = False)[0:7]

fig = px.pie(infection_cases, values = 'confirmed', names = 'infection_case', title = "Infection cases with most number of confirmed cases")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# <a id = 'sec-2d'> </a>
# ### What percentage are Group infections?

# In[ ]:


groups = case_data.groupby('group')['confirmed'].sum().reset_index()

fig = px.pie(groups, values='confirmed', names='group', title='Percentages of Group infections')
fig.update_traces(rotation=90, pull=0.05, textinfo='percent+label')
fig.show()


# <a id='sec-three'> </a>
# ## EDA of Patient Info
# 1. [Patient Info Data Information: What do we know?](#sec-3a)
# 2. [Are males more susceptible to the Covid-19 virus?](#sec-3b)
# 3. [Which Age group is more susceptible to the Covid-19 virus?](#sec-3c)
# 4. [Which Age group has the most deceased?](#sec-3d)
# 5. [Which Age group has the most recoveries?](#sec-3e)
# 6. [Patient state and their Age groups](#sec-3f)

# <a id = "sec-3a"> </a>
# ### Patient Info Data Information: What do we know?

# In[ ]:


patient_info.info()


# <a id = "sec-3b"> </a>
# ### Are males more susceptible to the Covid-19 virus?

# In[ ]:


'''
For the sex column, around 200 rows had Null values. Rather than replacing them and causing a biased distribution, I have decided to
remove the Null values for now.
'''
patient_info = patient_info[patient_info['sex'].notna()]

sex_count = patient_info.groupby('sex')['patient_id'].count().reset_index()

fig =  px.pie(sex_count, names='sex', values='patient_id', title='Confirmed Cases among Males and Females')
fig.update_traces(pull=0.05, textinfo='percent+label')
fig.show()


# <a id = "sec-3c"> </a>
# ### Which Age group is more susceptible to the Covid-19 virus?

# In[ ]:


'''
Infection rates are higher among the 20s then followed by the 50s
'''
patient_info = patient_info[patient_info['age'].notna()]

age_count = patient_info.groupby('age')['patient_id'].count().reset_index()

fig =  px.line(age_count, x="age", y="patient_id", title='Confirmed Cases among various Age groups')
fig.show()


# <a id = "sec-3d"> </a>
# ### Which Age group has the most deceased?

# In[ ]:


'''
Only 62 deceased dates have been recorded. Among that, 50+ seems to be the most affected age groups.
'''
deceased_count = patient_info.groupby('age')['deceased_date'].count().reset_index()

fig =  px.line(deceased_count, x="age", y="deceased_date", title='Age Groups and Deceased numbers')
fig.show()


# <a id = "sec-3e"> </a>
# ### Which Age group has the most recoveries?

# In[ ]:


'''
Under the assumption that the release date = recovery, this plot is drawn. Most recoveries are seen among the 20s 
followed by the 50s
'''
recovery_count = patient_info.groupby('age')['released_date'].count().reset_index()

fig =  px.line(recovery_count, x="age", y="released_date", title='Age Groups and Recovered numbers')
fig.show()


# <a id = "sec-3f"> </a>
# ### Patient state and their Age groups

# In[ ]:


patient_state = patient_info.groupby(['state', 'age'])['patient_id'].count().reset_index()
patient_state = patient_state.sort_values('age')

fig = px.line(patient_state, x='age', y='patient_id', color='state') 
fig.show()

