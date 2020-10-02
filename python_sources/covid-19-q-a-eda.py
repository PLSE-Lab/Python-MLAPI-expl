#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pygal')
get_ipython().system('pip install pygal_maps_fr')
get_ipython().system('pip install pygal_maps_world')
get_ipython().system('pip install lmfit')


# There are so many questions about Covid-19 such as which population are more likly to contract the virus?
# Whicch sex and age is overcoming this disease?
# To response to those quetions and many more we are going to follow steps below.
# 
# >***Import required libraries and datasets***
# 
# >***Are age /or sex can be risk factor?***
# 
# >***Are there more not discovered cases***
# 
# >***which french department is more exposed***
# 
# >***Import required libraries and datasets***

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from lmfit.model import Model
px.defaults.height = 400
ddir ="/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-cases.csv"
df =  pd.read_csv(ddir)
col = ['travel_history_country','locally_acquired','additional_info',
       'additional_source','additional_source','case_source','method_note']
ddir1 ="/kaggle/input/uncover/UNCOVER/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv"
ddir2 ="/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-mortality.csv"
ddir3 ="/kaggle/input/uncover/UNCOVER/github/covid19-epidemic-french-national-data.csv"
df3 =  pd.read_csv(ddir3)
df2 =  pd.read_csv(ddir2)
df1 =  pd.read_csv(ddir1)

df.drop(col,axis=1,inplace=True)
df.head()


# >***Are age /or sex can be risk factor?***
# 
# We  are comparing age and sex of confirmed cases. If we look at the graph below we can see clearly that sex is not rrisk factor the virus is attacking both Male and Female in th sme level.

# In[ ]:


layout = Layout(
    paper_bgcolor='black',
    plot_bgcolor='black'
)
sex_h = df[df['sex']=='Male'] 
sex_f = df[df['sex']=='Female'] 
fig = go.Figure(layout=layout)
fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),
                     y=sex_h['age'].value_counts(),name="Male",marker_color='yellow',
                    opacity=0.5)
                      )
fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),
                     y=sex_f['age'].value_counts(),name='Female',marker_color='lightgreen',
                    opacity=0.5)
                      )
fig.update_layout(barmode='relative', title_text='comparing ages of confirmed cases based on sex')
fig.show()


# In contrast, old peolple are more infected, but why?
# 
# - Perhaps they developed symptoms more than young people quicker 
# - Or they are benefiting from available tests more than young's , only doctors can confirm the hypothesis

# In[ ]:


fig = go.Figure(data=[go.Pie(labels=df['age'].value_counts().keys().drop("Not Reported"),
                             values=df['age'].value_counts().drop("Not Reported"),
                             title='Ages of confirmed cases',
                             hole=.3,pull=[0.2])],layout=layout)
fig.show()


# >***Are there more not discovered cases***
# 
# Some of the basic questions about this virus is do  confirmed cases give us  the real picture of the situation.
# To be able to answre to this question let's look at the situation in US particularly in New York. 

# In[ ]:


import plotly.express as px
px.defaults.color_continuous_scale = px.colors.plotlyjs
px.defaults.template = "ggplot2"
fig = px.scatter(df1,x='totaltestresults', 
                 y='positive',
                 color="state",size='total',marginal_x='histogram'
)
fig.show()


# By looking at the relationship between number of tests and positive cases we can see a strong positive correlation tests ans caces which means that there more cases that not yet.
# And if look at the curve at looks like exponential curve. 

# In[ ]:


df_ny = df1[df1['state']=='NY']
x = []
for i in range(len(df_ny.index)):
    x.append(i)
y = df_ny['positive']
def exp_func(x,a,b):
    return a*np.exp(b*x)
exponmodel = Model(exp_func)
params = exponmodel.make_params(a=5, b=0.01)
result = exponmodel.fit(y, params, x=x)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ny['date'], y=df_ny['positive'],line_color='rgb(231,107,243)',
    name='Positive cases',fill='tonexty'))
fig.add_trace(go.Scatter(x=df_ny['date'], y=result.best_fit,line_color='yellow',opacity=0.1,
    name='Positive exponential',fill='tozeroy'))
fig.update_layout(
Layout(
    paper_bgcolor='black',
    plot_bgcolor='black'),title_text='Fitting exponential curve to  positive cases ')

fig.update_traces(mode='lines')
fig.show()


# Now what about mortality?
# 
# This data reveal that ages is a high risk factor of mortality as we can observe in the figure below.

# In[ ]:


fig = px.pie(values=df2['age'].value_counts().drop("Not Reported"), 
             names=df2['age'].value_counts().keys().drop("Not Reported"),
             title='',hole=.3)
fig.update_layout(
Layout(
    paper_bgcolor='black',
    plot_bgcolor='black'),title_text='Ages of  death cases ')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# But oubviously it doesn't matter if you are Male or Female this virus can be fatal.  

# In[ ]:


layout = Layout(
    paper_bgcolor='black',
    plot_bgcolor='black'
)
sex_h = df[df['sex']=='Male'] 
sex_f = df[df['sex']=='Female'] 
fig = go.Figure(layout=layout)
fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),
                     y=sex_h['age'].value_counts(),name="Male",marker_color='purple',
                    opacity=0.5)
                      )
fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),
                     y=sex_f['age'].value_counts(),name='Female',marker_color='orange',
                    opacity=0.5)
                      )
fig.update_layout(barmode='relative', title_text='comparing ages of death cases based on sex')
fig.show()


# >***which french department is more exposed***
# 
# 
# Now in France I"am wondering which department has recorded high mortality rate by the end of the last month?

# In[ ]:


dffr = df3[df3['date']=='2020-03-30']
dffr[dffr['maille_code']=='FRA']
dffr = dffr[:101]
values = []
for v in dffr['deces']:
    values.append(v)
keys = []  
for k in dffr['maille_code']:
    keys.append(k.split('-')[-1])
res = {keys[i]: int(values[i]) for i in range(len(keys))}   
from IPython.display import HTML
import pygal
html_pygal = u"""
    <!DOCTYPE html>
    <html>
        <head>
            <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>
            <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/pygal-tooltips.js"></script>
        </head>
        <body><figure>{pygal_render}</figure></body>
    </html>
"""
from pygal.maps.fr import aggregate_regions
fr_chart = pygal.maps.fr.Departments(human_readable=True)
fr_chart.title = 'Covid-19 mortality by French department'
fr_chart.add('In 2020-03-30',res)
#fr_chart.render()
HTML(html_pygal.format(pygal_render=fr_chart.render(is_unicode=True)))


# ***If you are in Ile-de-France ,Bas-Rhin or Haut-Rhin  you should be more vigilant.***
