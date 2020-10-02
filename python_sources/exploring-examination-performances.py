#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ### Check for Null Values

# In[ ]:


df.isnull().any()


# In[ ]:


plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull())


# ### There has been a Recent Increase in the number of female students

# In[ ]:


trace1 = go.Bar(
            x=df['gender'].value_counts().index,
            y=df['gender'].value_counts().values,
            marker = dict(
                  line=dict(color='rgb(0,0,255)',width=2)),
            name = 'Weapons Acquired'
    )

data = [trace1]

layout = dict(title = 'Gender Distribution',
              xaxis= dict(title= 'Gender',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)


# ### Percentage Distribution

# In[ ]:


df['gender'].value_counts()/len(df)*100


# In[ ]:


trace1 = go.Bar(
            x=df['race/ethnicity'].value_counts().index,
            y=df['race/ethnicity'].value_counts().values,
            marker = dict(
                  line=dict(color='rgb(0,0,255)',width=2)),
            name = 'Cast and Creed'
    )

data = [trace1]

layout = dict(title = 'Cast/Races Distribution',
              xaxis= dict(title= 'Races',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)


# In[ ]:


temp = df['race/ethnicity'].value_counts()/len(df)*100


# ### Percentage Distribution - Races Ethnicity

# In[ ]:


labels = list(temp.index)
values = list(temp.values)
colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1', '#379A56']

trace = go.Pie(labels=labels, values=values,
              marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
data = [trace]
layout = dict(title = 'Cast/Races Percentage Distribution',
              xaxis= dict(title= 'Races',ticklen= 5),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)


# In[ ]:


colors = ['#379A56', '#D0F9B1']

trace1 = go.Bar(
            x=df['test preparation course'].value_counts().index,
            y=df['test preparation course'].value_counts().values,
            marker = dict(
                color=colors,
                  line=dict(color='rgb(0,0,0)',width=2)),
            name = 'Test Prepration'
    )

data = [trace1]

layout = dict(title = 'Test Prepration',
              xaxis= dict(title= 'Test Prepration',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Count")
             )
fig = dict(data = data, layout=layout)
iplot(fig)


# In[ ]:


df.columns


# In[ ]:


df['math score'].unique()


# ## Mathematics

# In[ ]:


fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(10)
fig.set_figwidth(10)

sns.pointplot(y='math score', x='race/ethnicity', data=df, ax=ax1)

sns.pointplot(y='math score', x='gender', data=df, ax=ax2)

sns.pointplot(y='math score', x='race/ethnicity', data=df, hue='test preparation course', ax=ax3)

sns.pointplot(y='math score', x='gender', hue='test preparation course', data=df, ax=ax4)


# ### *The above plot shows that male students and the Group E race dominates the maths community overall*
# ### *Also Test Preparation Course always improves the performance of Students*

# In[ ]:


fig, axarr = plt.subplots(1, 2)
ax1, ax2 = axarr[0], axarr[1]
fig.set_figheight(10)
fig.set_figwidth(18)

sns.pointplot(y='math score', x='parental level of education', data=df, ax=ax1)

sns.pointplot(y='math score', x='lunch', data=df, ax=ax2)


# ## Reading Score

# In[ ]:


fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(10)
fig.set_figwidth(10)

sns.pointplot(y='reading score', x='race/ethnicity', data=df, ax=ax1)

sns.pointplot(y='reading score', x='gender', data=df, ax=ax2)

sns.pointplot(y='reading score', x='race/ethnicity', data=df, hue='test preparation course', ax=ax3)

sns.pointplot(y='reading score', x='gender', hue='test preparation course', data=df, ax=ax4)


# In[ ]:


fig, axarr = plt.subplots(1, 2)
ax1, ax2 = axarr[0], axarr[1]
fig.set_figheight(10)
fig.set_figwidth(18)

sns.pointplot(y='reading score', x='parental level of education', data=df, ax=ax1)

sns.pointplot(y='reading score', x='lunch', data=df, ax=ax2)


# ### Similar Patterns Observed
# #### Females are comparatively proficient in reading as compared to the males for the given data whereas Males are dominant in Maths

# ## Writing Score

# In[ ]:


fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(10)
fig.set_figwidth(10)

sns.pointplot(y='writing score', x='race/ethnicity', data=df, ax=ax1)

sns.pointplot(y='writing score', x='gender', data=df, ax=ax2)

sns.pointplot(y='writing score', x='race/ethnicity', data=df, hue='test preparation course', ax=ax3)

sns.pointplot(y='writing score', x='gender', hue='test preparation course', data=df, ax=ax4)


# In[ ]:


fig, axarr = plt.subplots(1, 2)
ax1, ax2 = axarr[0], axarr[1]
fig.set_figheight(10)
fig.set_figwidth(18)

sns.pointplot(y='writing score', x='parental level of education', data=df, ax=ax1)

sns.pointplot(y='writing score', x='lunch', data=df, ax=ax2)


# # More Coming Soon
