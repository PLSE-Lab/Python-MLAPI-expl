#!/usr/bin/env python
# coding: utf-8

# # Plotly Visualization of Powerlifters

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/openpowerlifting.csv')
df.head(2)


# **Male vs Female distribution in the dataset**

# In[ ]:


labels = ['Male','Female']
colors = ['#1e90ff', '#E1396C']
gender = df['Sex'].value_counts()
values = list(gender.values)

trace = go.Pie(labels=labels, values=values,hoverinfo='label+percent',marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

py.iplot([trace], filename='gender_chart')


# In[ ]:


age = df['Age'].value_counts()
x = age.index
y = age.values

layout = go.Layout(
    title='Age distribution of Powerlifters',
    xaxis=dict(
        title='Age',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Number of Powerlifters',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

data = [go.Bar(
            x=x,
            y=y
    )]
py.iplot(go.Figure(data=data, layout=layout))


# ****Most of the powerlifters are in their 20s and 30s......Looks like there is a 90 year old powerlifter and also a  5 year old ****

# ** Distribution of Bodyweight of Powerlifters**

# In[ ]:


val = df['BodyweightKg'].value_counts()
trace1 = go.Scatter(
    x = val.index,
    y = val.values,
    mode='markers',
    marker=dict(
        size=16,
        color = val.values, #set color equal to a variable
        colorscale='Viridis',
        showscale=True
    )
)

layout = go.Layout(
    title='Weight distribution of Powerlifters',
    xaxis=dict(
        title='Weight in Kg',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Number of Powerlifters',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

data = [trace1]

py.iplot(go.Figure(data=data, layout=layout))


# In[ ]:


classes = df['WeightClassKg'].value_counts()
div = df['Division'].value_counts()
classes = classes[classes.values > 6260]
div = div[div.values > 2000]


# In[ ]:


fig = {
  "data": [
    {
      "values": classes.values,
      "labels": classes.index,
      "domain": {"x": [0, .48]},
      "name": "Weight Class in Kg",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": div.values,
      "labels": div.index,
      "text":["CO2"],
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "Division",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Weight Classes & Division Distribution of Powerlifters",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Classes",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Division",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='donut')



      


# - **About 9.4 % of the powerlifters seem to belong to the 90kg category followed by 8.95% in the 100 kg category**
# -  **About 30% of the powerlifters belong to the Open Division followed by about 25% in the Boys Division**

# ## The Equipments used by Powerlifters

# In[ ]:


eq = df['Equipment'].value_counts()
values = eq.values
phases = eq.index
colors = ['rgb(32,155,160)', 'rgb(253,93,124)', 'rgb(28,119,139)', 'rgb(182,231,235)', 'rgb(35,154,160)']
n_phase = len(phases)
plot_width = 400

# height of a section and difference between sections 
section_h = 100
section_d = 10

# multiplication factor to calculate the width of other sections
unit_width = plot_width / max(values)

# width of each funnel section relative to the plot width
phase_w = [int(value * unit_width) for value in values]

# plot height based on the number of sections and the gap in between them
height = section_h * n_phase + section_d * (n_phase - 1)

# list containing all the plot shapes
shapes = []

# list containing the Y-axis location for each section's name and value text
label_y = []

for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[i],
                'line': {
                    'width': 1,
                    'color': colors[i]
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)
# For phase names
label_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=phases,
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)
 
# For phase values
value_trace = go.Scatter(
    x=[350]*n_phase,
    y=label_y,
    mode='text',
    text=values,
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)

data = [label_trace, value_trace]
 
layout = go.Layout(
    title="<b>Equipments used by Powerlifters</b>",
    titlefont=dict(
        size=20,
        color='rgb(203,203,203)'
    ),
    shapes=shapes,
    height=560,
    width=900,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)
 
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)



# **Looks like Most of the powerlifters don't use equipments (RAW)**
# 
# - Single-ply suits/shirts have one layer of material.
# 
# - Multiply suits/shirts have multiple layers of material.
# - Powerlifters also use straps and wraps
# 

# ### Wilks Coefficient:
# 
# The Wilks Coefficient or Wilks Formula is a coefficient that can be used to measure the strength of a powerlifter against other powerlifters despite the different weights of the lifters. Robert Wilks, CEO of Powerlifting Australia, is the author of the formula.
# It is the total weight lifted (in kg) is multiplied by the Coefficient to find the standard amount lifted normalised across all body weights.
# 

# In[ ]:


ag = df.groupby(['Age']).mean()
wt =  df.groupby(['BodyweightKg']).mean()


# In[ ]:


data = [
    {
        'x': ag.index,
        'y': ag['Wilks'],
        'mode': 'lines+markers',
        'name': 'Wilks-Coefficient',
    }
]

layout = dict(title = 'Wilks Coefficient Distribution with respect to the Age of Powerlifters',
              yaxis = dict(title = 'Wilks Coefficient', zeroline = False),
              xaxis = dict(title= 'Age',zeroline = False)
             )
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='scat')


# 

# The Wilks Coefficient increases as age increases(It peaks in their 20s and 30s) and decreases as powerlifters become old which is normal and predictable......But it also looks like there is a 93 year old powerlifter with a high wilks coefficient....young by mind and body....quite fascinating

# #### Wilks: 
# - 300 is good
# - 400 is great.
# - 500 is elite.
# 
# 

# In[ ]:


above = wt[wt['Wilks'] > 300]
below = wt[wt['Wilks'] < 300]


trace0 = go.Scatter(
    x = above.index,
    y = above['Wilks'],
    name = 'Above 300',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(152, 0, 0, .8)',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x = below.index,
    y = below['Wilks'],
    name = 'Below 300',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2,
        )
    )
)

data = [trace0, trace1]

layout = dict(title = 'Wilks Coefficient Distribution with respect to the Weight of Powerlifters',
              yaxis = dict(title = 'Wilks Coefficient', zeroline = False),
              xaxis = dict(title= 'Weight in Kg',zeroline = False)
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')


# A lot of Powerlifters between of 60 - 180 kg seem to have an above average Wilks Coefficient whereas as the weight increases or decreases normal levels...wilks also decreases

# In[ ]:


pl =  df.groupby(['Place']).mean()
data = [
    {
        'x': pl.index,
        'y': pl['Wilks'],
        'mode': 'markers',
        'name': 'Wilks-Coefficient',
    }
]

layout = dict(title = 'Wilks Coefficient Distribution with respect to the Place of Powerlifters',
              yaxis = dict(title = 'Wilks Coefficient', zeroline = False),
              xaxis = dict(title= 'Place',zeroline = False)
             )
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='scat')


# ***Woah Woah.....Looks like Place 1 Powerlifters have a low mean Wilks coefficient....Weird and interesting ***

# ## Total of weights lifted vs Wilks of Powerlifters**

# In[ ]:


data = [
    {
        'x': df['TotalKg'],
        'y': df['Wilks'],
        'mode': 'markers',
        'name': 'Wilks-Coefficient',
    }
]

layout = dict(title = 'Wilks Coefficient Distribution with respect to the Place of Powerlifters',
              yaxis = dict(title = 'Wilks Coefficient', zeroline = False),
              xaxis = dict(title= 'Place',zeroline = False)
             )
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='scat')


# ### Total weight in kg and Wilks have an almost linear relationship......Makes Sense

# In[ ]:


from plotly import tools
deadlift = go.Scatter(
    x = ag.index,
    y = ag['BestDeadliftKg'],
    mode='markers',
    name='Deadlift'
)
Squat = go.Scatter(
    x = ag.index,
    y = ag['BestSquatKg'],
    mode='markers',
    name='Squat'

)
Bench = go.Scatter(
    x = ag.index,
    y = ag['BestBenchKg'],
    mode='markers',
    name='Bench'
)

Total = go.Scatter(
    x = ag.index,
    y = ag['TotalKg'],
    mode='markers',
    name='Total'
 
)

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Max Deadlift', 'Max Squat',
                                                          'Max Benchpress','Total Kgs'))

fig.append_trace(deadlift, 1, 1)
fig.append_trace(Squat, 1, 2)
fig.append_trace(Bench, 2, 1)
fig.append_trace(Total, 2, 2)


fig['layout'].update(height=800, width=1000, title='Age of Powerlifters' +
                                                  ' with respect to parameters')

py.iplot(fig, filename='sub')


# The Graphs seem to look very much similar.....but I couldnt help but notice an anamoly as the powerlifters get older
# 
# ### An Exceptionally Strong 93 year powerlifter.....Who could that be ?

# In[ ]:


df[(df['Age'] == 93)]


# ## Take a bow Dan Mason

# In[ ]:





# In[ ]:




