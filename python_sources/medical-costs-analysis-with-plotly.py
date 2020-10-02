#!/usr/bin/env python
# coding: utf-8

# <h1>INTRODUCTION</h1>
# 
# <br>Content:<br>
# 1. [Rate of Males & Females ](#1) <br>
# 2. [Body Mass Index vs Medical Costs Means (Low-Normal-High Values)](#2) <br>
# 3. [Female and Male's Body Mass Index ](#3) <br>
# 4. [The Most Medical Costs (Top 250) and Their Body Mass Index](#4) <br>
# 5. [Medical Costs Means by Regions](#5) <br>
# 6. [Rate of Smokers & Non-Smokers](#6) <br>
# 7. [Medical Costs of Smoker vs Non-Smokers](#7) <br>
# 8. [Chdildrens of Smokers and Non-Smokers](#8) <br>
# 
# <br> Information About Data:<br>
# * age: age of primary beneficiary 
# * sex: insurance contractor gender, female, male
# * bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# * children: Number of children covered by health insurance / Number of dependents
# * smoker: Smoking
# * region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# * charges: Individual medical costs billed by health insurance
# 
# 
# 
# <br> Visualization Libraries:<br>
# * Pandas - Data processing, data cleaning
# * Plotly - Data visualization
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
#os
import os
print(os.listdir("../input"))


# ## Dataset Overview

# In[ ]:


df = pd.read_csv("../input/insurance.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.region.unique()


# <a id="1"></a>
# ## 1. Rate of Males & Females

# Preparing data.

# In[ ]:


gender_list = [df[df.sex == "female"].sex.value_counts().tolist(), df[df.sex == "male"].sex.value_counts().tolist()]
gender_list = [gender_list[0][0], gender_list[1][0]]
gender_list


# In[ ]:


labels = ["Female", "Male"]
values = gender_list
colors = ['#FEBFB3', '#b3c8fe']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='percent', 
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
data = [trace]
layout = go.Layout(title='Rate of Males & Females')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="2"></a>
# ## 2. Body Mass Index vs Medical Costs Means (Low-Normal-High Values)

# Cleaning and preparing data.

# In[ ]:


dict_regions= {'low' : df[df.bmi < 18.5].charges.mean(),
               'normal' : df[(df.bmi > 18.5) & (df.bmi < 24.9)].charges.mean(),
               'high' : df[df.bmi > 24.9].charges.mean(),
             }
df_bmi = pd.DataFrame.from_dict(dict_regions, orient='index')
df_bmi.reset_index(inplace=True)
df_bmi.columns = ['bmi', 'mean_value']
df_bmi


# In[ ]:


my_color = ['rgb(254,224,39)','rgb(102,189,99)','rgb(215,48,39)']
trace=go.Bar(
            x=df_bmi.bmi,
            y=df_bmi.mean_value,
            text="Mean Medical Costs",
            marker=dict(
                color=my_color,
                line=dict(
                color=my_color,
                width=1.5),
            ),
            opacity=0.7)

data = [trace]
layout = go.Layout(title = 'Body Mass Index Means',
              xaxis = dict(title = 'BMI'),
              yaxis = dict(title = 'Mean Charges'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# 

# <a id="3"></a>
# ## 3. Female and Male's Body Mass Index 

# In[ ]:


trace0 = go.Box(
    y=df[df.sex == "female"].bmi,
    name = 'Female',
    marker = dict(
        color = 'rgb(158, 1, 66)',
    )
)
trace1 = go.Box(
    y=df[df.sex == "male"].bmi,
    name = 'Male',
    marker = dict(
        color = 'rgb(50, 136, 189)',
    )
)
layout = go.Layout(title ='BMI of Females and Males',
              xaxis = dict(title = 'Gender'),
              yaxis = dict(title = 'BMI'))
data = [trace0, trace1]
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="4"></a>
# ## 4. The Most Medical Costs (Top 250) and Their Body Mass Index)

# Preparing data.

# In[ ]:


charges_sorted = df.copy()
sort_index = (df['charges'].sort_values(ascending=False)).index.values
charges_sorted = df.reindex(sort_index)
charges_sorted.reset_index(inplace=True)
#charges_sorted = charges_sorted.head(250)
charges_sorted.head()


# In[ ]:


# bmi values above-below
trace0 = go.Scatter(
    x = charges_sorted.head(250).charges,
    y = charges_sorted.head(250).bmi[charges_sorted.head(250).bmi < 18.5],
    name = 'Low',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(254,224,39)',
        line = dict(
            width = 1,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x = charges_sorted.head(250).charges,
    y = charges_sorted.head(250).bmi[(charges_sorted.head(250).bmi > 18.5) & (charges_sorted.head(250).bmi < 24.9)],
    name = 'Normal',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(102,189,99)',
        line = dict(
            width = 1,
        )
    )
)

trace2 = go.Scatter(
    y = charges_sorted.head(250).bmi[charges_sorted.head(250).bmi > 24.9],
    x = charges_sorted.head(250).charges,
    name = 'High',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(215,48,39)',
        line = dict(
            width = 1,
        )
    )
)
data = [trace0, trace1,trace2]
layout = dict(title = 'BMI of the Most 250 Medical Costs',
              yaxis = dict(zeroline = False,title = "BMI"),
              xaxis = dict(zeroline = False,title = "Medical Cost"),
             )
fig = go.Figure(data = data, layout = layout)

iplot(fig)


# <a id="5"></a>
# ## 5. Medical Costs Means by Regions

# Cleaning and preparing data.

# In[ ]:


dict_regions= {'southwest' : df[df.region == "southwest"].charges.mean(),
              'southeast' : df[df.region == "southeast"].charges.mean(),
              'northwest' : df[df.region == "northwest"].charges.mean(),
              'northeast' : df[df.region == "northeast"].charges.mean()
             }
df_regions = pd.DataFrame.from_dict(dict_regions, orient='index')
df_regions.reset_index(inplace=True)
df_regions.columns = ['regions', 'charges']

df_regions


# In[ ]:


import plotly.graph_objs as go
import colorlover as cl

trace=go.Bar(
            x=df_regions.regions,
            y=df_regions.charges,
            text="Mean Medical Costs",
            marker=dict(
                color=cl.scales['12']['qual']['Paired'],
                line=dict(
                color=cl.scales['12']['qual']['Paired'],
                width=1.5),
            ),
            opacity=0.8)

data = [trace]
layout = go.Layout(title ='Medical Cost Means by Regions',
              xaxis = dict(title = 'Region'),
              yaxis = dict(title = 'Medical Cost'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="6"></a>
# ## 6. Rate of Smokers & Non-Smokers

# Preparing data.

# In[ ]:


smoker_list = [df[df.smoker == "yes"].smoker.value_counts().tolist(), df[df.smoker == "no"].smoker.value_counts().tolist()]
smoker_list = [smoker_list[0][0], smoker_list[1][0]]
smoker_list


# In[ ]:


labels = ["Smoker", "Non-Smoker"]
values = smoker_list
colors = ['#feb3b3', '#c5feb3']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='percent', 
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
data = [trace]
layout = go.Layout(title='Rate of Smokers & Non-Smokers')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="7"></a>
# ## 7. Medical Costs of Smoker vs Non-Smokers

# Let's recall out sorted data by charges. But this time, using all data instead of first 250.

# In[ ]:


charges_sorted.head()


# In[ ]:


trace0 = go.Scatter(
    x = charges_sorted.index,
    y = charges_sorted[charges_sorted.smoker == "yes"].charges,
    name = "Smokers",
    mode='lines',
    marker=dict(
        size=12,
        color = "red", #set color equal to a variable
    )
)

trace1 = go.Scatter(
    x = charges_sorted.index,
    y = charges_sorted[charges_sorted.smoker == "no"].charges,
    name = "Non-Smokers",
    mode='lines',
    marker=dict(
        size=12,
        color = "green", #set color equal to a variable
    )
)


data = [trace0,trace1]
layout = go.Layout(title = 'Medical Costs of Smoker vs Non-Smokers',
              xaxis = dict(title = 'Persons'),
              yaxis = dict(title = 'Medical Costs'),)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="8"></a>
# ## 8. Chdildrens of Smokers and Non-Smokers

# Cleaning and preparing data.

# In[ ]:


df_smokers = df[df.smoker == "yes"]
df_smokers.reset_index(inplace=True)
df_non_smokers = df[df.smoker == "no"]
df_non_smokers.reset_index(inplace=True)


# In[ ]:


trace0 = go.Histogram(
    x=df_non_smokers.children,
    opacity=0.75,
    name = "Non-Smokers",
    marker=dict(color='rgba(166, 217, 106, 1)'))

trace1 = go.Histogram(
    x=df_smokers.children,
    opacity=0.75,
    name = "Smokers",
    marker=dict(color='rgba(244, 109, 67, 1)'))

data = [trace0,trace1]
layout = go.Layout(barmode='overlay',
                   title='Childrens of Smokers vs Non-Smokers',
                   xaxis=dict(title='Number of Children'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Conclusion
# * This is being my first data analysis with Plotly.
# * I have tried to use different plots and make practice about Plotly data visualization.
# * Please comment section and upvote if you liked my kernel, thank you.
