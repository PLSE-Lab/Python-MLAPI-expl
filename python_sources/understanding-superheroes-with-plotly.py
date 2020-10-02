#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


marvel = pd.read_csv('/kaggle/input/comic-characters/marvel-wikia-data.csv')
dc = pd.read_csv('/kaggle/input/comic-characters/dc-wikia-data.csv')


# In[ ]:


marvel.head()


# In[ ]:


dc.head()


# # Plotting between Appearances, Year and Alignment:

# In[ ]:


import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

Marvel_Points = go.Scatter3d(
    x = dc['APPEARANCES'],
    y = dc['YEAR'],
    z = dc['ALIGN'],
    name = 'DC',
    mode='markers',
    marker=dict(
        size=10,
        color = 'purple'      
    )
)

DC_Points = go.Scatter3d(
    x = marvel['APPEARANCES'],
    y = marvel['Year'],
    z = marvel['ALIGN'],
    name = 'Marvel',
    mode = 'markers',
    marker = dict(
         size = 10,
         color = 'yellow'
    )
)

data = [Marvel_Points, DC_Points]

layout = go.Layout(
    title = 'Character Appearances vs. Year vs. Alignment',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# # Appearances of Marvel characters over the years:

# In[ ]:


trace = go.Bar(
    x=marvel['Year'],
    y=marvel['APPEARANCES'],
    marker=dict(
        color='red',
        colorscale = 'Reds')
)

data = [trace]
layout = go.Layout(
    title='Apperances of Marvel heroes throughout the years', 
    yaxis = dict(title = 'Appearances')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **1963 seems to have been a great year for Marvel characters to come on the stage.**

# # Appearances of DC characters over the years:

# In[ ]:


trace = go.Bar(
    x=dc['YEAR'],
    y=dc['APPEARANCES'],
    marker=dict(
        color='red',
        colorscale = 'Reds')
)

data = [trace]
layout = go.Layout(
    title='Apperances of DC heroes throughout the years', 
    yaxis = dict(title = 'Appearances')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **DC characters appear more consistently across the years, after peaking in 1940.**

# # Eye Color Distribution in Marvel Characters:

# In[ ]:


def PieChart(df, column, title, limit):
    
    male = df[df['SEX'] == 'Male Characters']
    female = df[df['SEX'] == 'Female Characters']
    count_male = male[column].value_counts()[:limit].reset_index()
    count_female = female[column].value_counts()[:limit].reset_index()
    color = ['red',  'navy',  'cyan', 'lightgrey','orange', 'gold','lightgreen', 
                            '#D0F9B1','tomato', 'tan']
    
    trace1 = go.Pie(labels=count_male['index'], 
                    values=count_male[column], 
                    name= "male", 
                    hole= .5, 
                    domain= {'x': [0, .48]},
                   marker=dict(colors=color))

    trace2 = go.Pie(labels=count_female['index'], 
                    values=count_female[column], 
                    name="female", 
                    hole= .5,  
                    domain= {'x': [.52, 1]})

    layout = dict(title= title, font=dict(size=15), legend=dict(orientation="h"),
                  annotations = [
                      dict(
                          x=.20, y=.5,
                          text='Male', 
                          showarrow=False,
                          font=dict(size=20)
                      ),
                      dict(
                          x=.81, y=.5,
                          text='Female', 
                          showarrow=False,
                          font=dict(size=20)
                      )
        ])

    fig = dict(data=[trace1, trace2], layout=layout)
    iplot(fig)


# In[ ]:


PieChart(marvel, 'EYE', "Eye Color Distribution", 6)


# **In Marvel characters, most males seem to have brown eyes, and females, blue eyes.**

# # Eye Color Distribution in DC Characters:

# In[ ]:


PieChart(dc, 'EYE', "Eye Color Distribution", 6)


# **In DC characters, both males and females have predominantly blue eyes, with brown coming a close second.**

# # Representation of GSM Characters in Marvel

# In[ ]:


trace = go.Bar(
    x=marvel['GSM'],
    y=marvel['Year'],
    marker=dict(
        color='red',
        colorscale = [[0, 'rgb(166,206,227)'], [0.25, 'rgb(31,120,180)'], 
                      [0.45, 'rgb(178,223,138)'], 
                      [0.65, 'rgb(51,160,44)'], [0.85, 'rgb(251,154,153)'], 
                      [1, 'rgb(227,26,28)']],
        reversescale = True))

data = [trace]
layout = go.Layout(
    title='Gender and Sexual Minorities', 
    yaxis = dict(title = '# of GSM Characters')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **Marvel seems to have featured homosexual characters(gays & lesbians) more than other GSMs.**

# # Representation of GSM Characters in DC

# In[ ]:


trace = go.Bar(
    x=dc['GSM'],
    y=dc['YEAR'],
    marker=dict(
        color='red',
        colorscale = [[0, 'rgb(166,206,227)'], [0.25, 'rgb(31,120,180)'], 
                      [0.45, 'rgb(178,223,138)'], 
                      [0.65, 'rgb(51,160,44)'], [0.85, 'rgb(251,154,153)'], 
                      [1, 'rgb(227,26,28)']],
        reversescale = True))

data = [trace]
layout = go.Layout(
    title='Gender and Sexual Minorities', 
    yaxis = dict(title = '# of GSM Characters')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **DC has also predominantly featured homosexuals, but their distribution of GSMs is lesser compared to Marvel's.**

# # Superhero identities across the years:

# In[ ]:


trace0 = go.Scatter(
    x=marvel.Year,
    y=marvel.ID,
    name = 'Marvel',
    mode='markers',
    marker=dict(size=11,
        color=('navy')
               )
)

trace1 = go.Scatter(
    x = dc.YEAR,
    y = dc.ID,
    name = 'DC',
    mode='markers',
    marker=dict(size=11,
        color = ('aqua')
               )
)

data = [trace0, trace1]
layout = dict(title = 'The distribution of superhero identities across the years',
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Identity')
             )

fig = dict(data=data, layout=layout)
fig['layout']['xaxis'].update(dict(title = 'Year', 
                                   tickmode='linear',
                                   tickfont = dict(size = 10)))
iplot(fig)


# **The clear majority of heroes have public and secret identities, and almost all DC heroes have dual identities.**

# **Thank you for reading. If you want to recreate these graphs yourselves, just copy and paste the code, and change the x, y values at the top.**
