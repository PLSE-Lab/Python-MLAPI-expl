#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/hospital-bed-capacity-and-covid19/HRR Scorecard_ 20 _ 40 _ 60 - 20 Population.csv")


# In[ ]:


data = data.drop(0)
data.head(10)


# In[ ]:


data.columns


# In[ ]:





# In[ ]:


data = data.replace(',','', regex=True)
data = data.replace('%','', regex=True)

data


# In[ ]:


data['Available Hospital Beds'].unique()


# In[ ]:


data1 = data.drop(['HRR'],axis=1)
data1 = data1.apply(pd.to_numeric)
data1


# In[ ]:


data2 = data['HRR']
data3 = pd.concat([data2, data1], axis=1, join='inner')
data = data3


# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Adult Population',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Current Adult(18+) Population in different Locations of US ')
fig.show()


# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Total Hospital Beds',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Total Available Beds in different Locations.')
fig.show()


# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Total ICU Beds',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Total Available Beds in different Locations.')
fig.show()


# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Percentage of Potentially Available Beds Needed, Six Months',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Percentage of Potentially Available Beds Needed in next 6 Months')
fig.show()


# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Percentage of Potentially Available Beds Needed, Twelve Months',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Percentage of Potentially Available Beds Needed in next 12 Months')
fig.show()


# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Percentage of Potentially Available Beds Needed, Eighteen Months',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Percentage of Potentially Available Beds Needed in next 18 Months')
fig.show()


# In[ ]:


fig1data = data.sort_values(by = ['Adult Population'],ascending = False).head(15)


# # Highest adult populations.

# In[ ]:


fig1 = px.pie(fig1data, values='Adult Population', names='HRR')
fig1.update_traces(rotation=90, pull=0.1, textinfo="value")

fig1.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig1.show()


# In[ ]:


fig2data = data.sort_values(by = ['Available Hospital Beds'],ascending='False').head(15)
fig2 = px.pie(fig1data, values='Available Hospital Beds', names='HRR')
fig2.update_traces(rotation=90, pull=0.1, textinfo="value")
fig2.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig2.show()


# In[ ]:


fig3data = data.sort_values(by = ['Available ICU Beds'],ascending='False').head(20)
fig3 = px.pie(fig1data, values='Available ICU Beds', names='HRR')
fig3.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig3.update_traces(rotation=90, pull=0.1, textinfo="value")

fig3.show()


# In[ ]:


fig3data = data.sort_values(by = ['Projected Infected Individuals'],ascending='False').head(20)
fig3 = px.pie(fig1data, values='Projected Infected Individuals', names='HRR')
fig3.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig3.update_traces(rotation=90, pull=0.1, textinfo="value")

fig3.show()


# # RATIO OF PROJECTED INFECTED INDIVUALS TO THE AVAILABLE BEDS

# In[ ]:


data['Ratio of Infected Individuals per Available beds'] = data['Projected Hospitalized Individuals'] / data['Available Hospital Beds']


# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Ratio of Infected Individuals per Available beds',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Higher the ratio, More chances of suuffering.')
fig.show()


#  # Locations with *Lowest* Hospital beds per projected infected individuals who need to be hospitalized.

# In[ ]:


fig5data = data.sort_values(by = ['Ratio of Infected Individuals per Available beds'],ascending=False).head(15)
fig5 = px.pie(fig5data, values='Ratio of Infected Individuals per Available beds', names='HRR')
fig5.update_traces(textposition='inside')
fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig5.update_traces(rotation=90, pull=0.1, textinfo="value")

fig5.show()


# In[ ]:


import seaborn as sns
fig_dims = (20, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x = fig5data['Ratio of Infected Individuals per Available beds'],ax=ax, y=fig5data['HRR'])


# # Locations with *Highest* Hospital beds per projected infected individuals who need to be hospitalized.

# In[ ]:


fig5data = data.sort_values(by = ['Ratio of Infected Individuals per Available beds'],ascending=True).head(15)
fig5 = px.pie(fig5data, values='Ratio of Infected Individuals per Available beds', names='HRR')
fig5.update_traces(textposition='inside')
fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig5.update_traces(rotation=90, pull=0.1, textinfo="value")

fig5.show()


# In[ ]:


import seaborn as sns
fig_dims = (20, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x = fig5data['Ratio of Infected Individuals per Available beds'],ax=ax, y=fig5data['HRR'])


# In[ ]:


data['Ratio of Individuals that need ICU care per Available ICU beds'] = data['Projected Individuals Needing ICU Care'] / data['Available ICU Beds']
data


# # RATIO OF THE INDIVIDUALS WHO NEED ICU CARE TO THE AVAILABLE ICU BEDS.

# In[ ]:


fig = px.treemap(data, path=['HRR'], values='Ratio of Individuals that need ICU care per Available ICU beds',
                  color='Adult Population', hover_data=['HRR'],
                  color_continuous_scale='dense', title='Higher the ratio, More chances of suuffering.')
fig.show()


# # Locations with *Lowest* ICU beds per projected infected individuals who need ICU care.

# In[ ]:


fig5data = data.sort_values(by = ['Ratio of Individuals that need ICU care per Available ICU beds'],ascending=False).head(15)
fig5 = px.pie(fig5data, values='Ratio of Individuals that need ICU care per Available ICU beds', names='HRR')
fig5.update_traces(textposition='inside')
fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig5.update_traces(rotation=90, pull=0.1, textinfo="value")

fig5.show()


# In[ ]:


import seaborn as sns
fig_dims = (20, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x = fig5data['Ratio of Individuals that need ICU care per Available ICU beds'],ax=ax, y=fig5data['HRR'])


# # # Locations with *Highest* ICU beds per projected infected individuals who need ICU care.

# In[ ]:


fig5data = data.sort_values(by = ['Ratio of Individuals that need ICU care per Available ICU beds'],ascending=True).head(15)
fig5 = px.pie(fig5data, values='Ratio of Individuals that need ICU care per Available ICU beds', names='HRR')
fig5.update_traces(textposition='inside')
fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')
fig5.update_traces(rotation=90, pull=0.1, textinfo="value")

fig5.show()


# In[ ]:


import seaborn as sns
fig_dims = (20, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x = fig5data['Ratio of Individuals that need ICU care per Available ICU beds'],ax=ax, y=fig5data['HRR'])


# # *  York PA tops the list with lowest ICU beds for infected covid19 patients.
# # While,East Long Island NY tops list with lowest hospital beds for infected covid19 patients.

# # Hence, We could conclude these two areas could face major threat in future. Please upvote if you liked the visuals. Thank you. 
