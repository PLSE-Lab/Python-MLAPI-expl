#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from collections import Counter
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns





from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Introcudtion
# Coronavirus is the one of the most important issues that the world faces nowadays and South Korea is very successful on handle it. In this kernel, I just want to visualize data about cases from South Korea so that may be we can understand what they did right.
# 
# <font color = 'green'>
# Content:
#     
# 1. [Distribution of cases by cities](#1)
# 2. [Group Interactions](#2)
# 3. [Patients Examinations](#3)
#    * [Number of cases by age](#3.1)
#    * [Distribution of cases over genders](#3.2)
# 4. [States of People](#5)
# 5. [First 21 days results](#6)
#    * [Negative - positive comparison](#6.1)
#    * [States of Confirmed People](#6.2) 
# 6. [Last 21 days results](#7)
#    * [Negative - positive comparison](#7.1)
#    * [States of Confirmed People](#7.2) 
# 

# In[ ]:


cases = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')
cases.info()


# <a id = "1"></a>
# ## Distribution of cases by cities

# In[ ]:



#cases.confirmed = cases.confirmed.astype(float)

province_list = list(cases['province'].unique())
number_of_cases = []
for j in province_list:
    x = cases[cases['province'] == j]
    total = sum(x.confirmed)
    number_of_cases.append(total)
data = pd.DataFrame({'province_list': province_list, 'number_of_cases' : number_of_cases})
new_index = (data['number_of_cases'].sort_values(ascending = False)).index.values
sorted_data = data.reindex(new_index)

#visualization
plt.figure(figsize = (10,5))
sns.barplot(x = sorted_data['province_list'], y = sorted_data['number_of_cases'])
plt.xticks(rotation = 80)
plt.xlabel('Provinces')
plt.ylabel('Number of Cases')
plt.title('Number of Cases for each province')



labels = cases.province.value_counts().index




#figure
fig = {
    "data" : [
        {
            "values" : number_of_cases,
            "labels" : labels,
            "domain" : {"x":[0, .5]},
            "name" : "Distrubition of Cases over cities",
            "hoverinfo" : "label+percent+name",
            "hole" : .3,
            "type" : "pie"
        },
    ],
    "layout" : {
        "title" : "Distrubiton of Cases Over Cities",
        "annotations" : [
            {
                "font" : {"size" : 20},
                "showarrow" : False,
                "text" : "Cases",
                "x" : 0.20,
                "y" : 1
            },
        ]
    }
}
iplot(fig)


# <a id = "2"></a>
# ## Let's see how dangerous group interaction is

# In[ ]:


cases.group.dropna(inplace = True)
labels = ['Group Infection','Not Group']
colors = ['red', 'green']
explode = [0,0]
sizes = cases.group.value_counts().values

#visualization
plt.figure(figsize = (8,8))
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%.1f%%')
plt.title('Group Infection Rates', color = 'black',fontsize = 12)
plt.show()


# <a id = 3></a>
# ## 3. Patients Examinations 

# In[ ]:


patients = pd.read_csv('../input/coronavirusdataset/PatientInfo.csv')
patients.head()


# In[ ]:


patients.info()


# <a id = 3.1></a>
#  ### * Number of cases by age

# In[ ]:


patients.age.dropna(inplace = True)
assert 1 == 1
ages = []
for x in patients.age:
    ages.append(x)

ages.sort()
age_counts = Counter(ages) 
p = age_counts.most_common(10)

x,y = zip(*p)
x,y = list(x),list(y)

plt.figure(figsize = (10,7))
ax = sns.barplot(x = x, y = y, palette = sns.cubehelix_palette(17))
plt.xlabel('Ages')
plt.ylabel('Counts')
plt.title('Number of cases by age')
plt.show()


# <a id = 3.2></a>
#  ### * Distribution of cases over genders

# In[ ]:


patients.sex.value_counts()
sns.countplot(patients.sex)
plt.title('Distribution of cases over genders', color = 'red', fontsize = 13)
plt.show()


# <a id = 5></a>
# ## 5. States of people

# In[ ]:


patients.state.dropna(inplace = True)
labels = ['Isolated','Released','Deceased']
colors = ['purple','green','red', ]
explode = [0,0,0]
sizes = patients.state.value_counts()

#visualization
plt.figure(figsize = (7,7))
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%.2f%%')
plt.title('States of People who exposed virus', color = 'black',fontsize = 14)
plt.show()


# In[ ]:


time_data = pd.read_csv('../input/coronavirusdataset/Time.csv')
time_data.info()


# <a id = 6></a>
# ## First 21 days results

# <a id = 6.1></a>
# ### * Negative - positive comparison

# In[ ]:


first_21_days = time_data.head(21)

trace1 = go.Scatter(
                   x = first_21_days.test,
                   y = first_21_days.negative,
                   mode = "lines",
                   name = "Negatives",
                   marker = dict(color = 'rgba(16,112,2,0.8)'),
                   text = first_21_days.date
                   )

trace2 = go.Scatter(
                   x = first_21_days.test,
                   y = first_21_days.confirmed,
                   mode = "lines + markers",
                   name = "Positives",
                   marker = dict(color = 'rgba(80,10,22,0.8)'),
                   text = first_21_days.date
                   )
data = [trace1, trace2]
layout = dict(title = 'Negative - positive comparison in the first 21 days', xaxis = dict(title = '# of negative or positive', ticklen = 5, zeroline = False), yaxis = dict(title = 'Number of tests'))
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id = 6.2></a>
# ### * States of Confirmed People in the first 21 days

# In[ ]:




f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=first_21_days.confirmed,y=first_21_days.date,color='blue',alpha = 0.5,label='Confirmed' )
sns.barplot(x=first_21_days.released,y=first_21_days.date,color='green',alpha = 0.7,label='Released')
sns.barplot(x=first_21_days.deceased,y=first_21_days.date,color='red',alpha = 0.6,label='Deceased')

ax.legend( loc = 'upper right', frameon = True)
ax.set(xlabel = 'States', ylabel = 'Days', title = 'States of Confirmed People')
plt.show()


# <a id = 7></a>
# ## Last 21 days results

# <a id = 7.1></a>
# ### * Negative - positive comparison

# In[ ]:


last_21_days = time_data.tail(21) 

trace1 = go.Scatter(
                   x = last_21_days.test,
                   y = last_21_days.negative,
                   mode = "lines",
                   name = "Negatives",
                   marker = dict(color = 'rgba(16,112,2,0.8)'),
                   text = last_21_days.date
                   )

trace2 = go.Scatter(
                   x = last_21_days.test,
                   y = last_21_days.confirmed,
                   mode = "lines + markers",
                   name = "Positives",
                   marker = dict(color = 'rgba(80,10,22,0.8)'),
                   text = last_21_days.date
                   )
data = [trace1, trace2]
layout = dict(title = 'Negative - positive comparison in the last 21 days', xaxis = dict(title = '# of negative or positive', ticklen = 5, zeroline = False), yaxis = dict(title = 'Number of tests'))
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id = 7.2></a>
# ### * States of Confirmed People in the last 21 days

# In[ ]:


f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=last_21_days.confirmed,y=last_21_days.date,color='blue',alpha = 0.3,label='Confirmed' )
sns.barplot(x=last_21_days.released,y=last_21_days.date,color='green',alpha = 0.7,label='Released')
sns.barplot(x=last_21_days.deceased,y=last_21_days.date,color='red',alpha = 0.9,label='Deceased')

ax.legend( loc = 'upper right', frameon = True)
ax.set(xlabel = 'States', ylabel = 'Days', title = 'States of Confirmed People')
plt.show()

