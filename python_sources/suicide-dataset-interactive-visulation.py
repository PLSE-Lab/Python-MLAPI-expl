#!/usr/bin/env python
# coding: utf-8

# # Set Up

# In[ ]:


# Data Cleaning & Manipulation Libraries
import numpy as np
import pandas as pd


# In[ ]:


# Data Visulation Libraries
import matplotlib.pyplot as plt
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()


# # Accessing Dataset with Pandas

# In[ ]:


# Reading master.csv file using Pandas
Suicide = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')


# In[ ]:


# Suicide Dataset info.
Suicide.info()     # Eagle Eye View


# In[ ]:


# Suicide Dateset 
Suicide.head()   # In-depth View


# # Data Visualized in this Notebook :
#     - Q1 : Total Population of each Country
#     - Q2 : World's Population as per Age Category
#     - Q3 : Total Suicides per Country
#     - Q4 : Total Suicides per Year
#     - Q5 : Total Suicides per Generation and Sex
#     - Q6 : Suicide rate per 100k people
#     - Q7 : Suicide rate per Age Category

# # **Q1 : Total Population of each Country**

# In[ ]:


# Data Set    
total_ppl = Suicide.groupby(by='country').population.sum().sort_values(ascending=False)
total_ppl     # 'United States with highest population' & 'Dominica with lowest population'


# In[ ]:


# Data Visulation - Bar Plot
total_ppl.iplot(kind='bar',title='Population across the Globe',xaxis_title='Countries',yaxis_title='Population in Billion')


# In[ ]:


# Geographical Visulation - choropleth map

# Data Object
data = dict(type='choropleth',
           locations = total_ppl.index,
           locationmode = 'country names',
           z = total_ppl[:],
           colorscale = 'oranges',
           colorbar = {'title':'Population in Billion'},
           text = total_ppl.index)
# Layout Object
layout = dict(geo = dict(projection={'type':'orthographic'},showframe=False),title='Population across the Globe')
# Plotting
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# # **Q2 : World's Population as per Age Category**

# In[ ]:


# Minor Change in age column     # for personal preference only
Suicide['age'].replace(to_replace='5-14 years',value='05-14 years',inplace=True)


# In[ ]:


# Data Set
Suicide.groupby('age').population.sum()


# In[ ]:


# Data Visulation - Bar Plot
Suicide.groupby('age').population.sum().iplot(kind='bar',title='World\'s Population as per Age Category',
                                              xTitle='Age Category',yTitle='Population in Billions')


# In[ ]:


# Data Visulation - Pie Chart
go.Figure(data=go.Pie(labels=Suicide.groupby('age').population.sum().index, values=Suicide.groupby('age').population.sum()[:],
                      title=' World\'s Population as per Age Category'))


# # **Q3 : Total Suicides per Country**

# In[ ]:


# Data Set
total_suicide = Suicide.groupby('country').suicides_no.sum().sort_values(ascending=False)
total_suicide    # 'Russia with highest suicides' & 'Dominica with lowest suicides'


# In[ ]:


# Data Visulation - Bar Plot
total_suicide.iplot(kind='bar',color='red',title='Suicides per Country',xTitle='Countries',yTitle='Suicide Counts')


# In[ ]:


# Data Visulation - Line Plot
total_suicide.iplot(kind='line',color='red',title='Suicides per Country',xTitle='Countries',yTitle='Suicide Counts')


# In[ ]:


# Geographical Visulation - choropleth map

# Data Object
data = dict(type='choropleth',
           locations = total_suicide.index,
           locationmode = 'country names',
           z = total_suicide[:],
           colorscale = 'viridis',
           colorbar = {'title':'Suicides in Million'},
           text = total_suicide.index)
# Layout Object
layout = dict(geo = dict(projection={'type':'orthographic'},showframe=False),title='Suicide around the World')
# Plotting
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# # **Q4 : Total Suicides per Year**

# In[ ]:


# dataset
Suicide.groupby('year').suicides_no.sum()


# In[ ]:


# Maximum no. of suicide & year of occurence
print('Maximum Suicide Number :',Suicide.groupby('year').suicides_no.sum().max())
print('Maximum Suicide Year :',Suicide.groupby('year').suicides_no.sum().idxmax())

# Minimum no. of suicide & year of occurence
print('\nMinimum Suicide Number :',Suicide.groupby('year').suicides_no.sum().min())
print('Minimum Suicide Year :',Suicide.groupby('year').suicides_no.sum().idxmin())


# In[ ]:


# Data Visulation - Bar Plot
Suicide.groupby('year').suicides_no.sum().iplot(kind='bar',title='Suicides per Year 1985 to 2016',
                                                xTitle='Year',yTitle='Suicide Counts')


# In[ ]:


# Data Visulation - Line Graph
Suicide.groupby('year').suicides_no.sum().iplot(kind='line',title='Suicides per Year 1985 to 2016',
                                                xTitle='Year',yTitle='Suicide Counts')


# # **Q5 : Total Suicides per Generation and Sex**

# In[ ]:


# Dataset
df = Suicide.groupby(['generation','sex']).suicides_no.sum().unstack()
df


# In[ ]:


# Data Visulation - Bar Plot
df.iplot(kind='bar',title='Suicides per Generation',xTitle='Generation',yTitle='Total Suicides')


# # **Q6 : Suicide rate per 100k people**

# In[ ]:


# Data Set
Suicide.groupby(['year','sex']).mean()['suicides/100k pop'].unstack()


# In[ ]:


# Data Visulation - Line Plot
Suicide.groupby(['year','sex']).mean()['suicides/100k pop'].unstack().iplot(kind='line',
                                title='Suicide Rate per 100K People',xTitle='Year',yTitle='Suicide Rate')


# # **Q7 : Suicide rate per Age Category**

# In[ ]:


# Data Set
Suicide.groupby('age').mean()['suicides/100k pop']


# In[ ]:


# Data Visulation - Pie Chart
go.Figure(data=[go.Pie(labels=Suicide['age'],values=Suicide['suicides/100k pop'],
                       title='Suicide Rate per Age Category')])

