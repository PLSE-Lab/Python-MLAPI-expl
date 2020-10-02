#!/usr/bin/env python
# coding: utf-8

# # Reddit r/ForeverAlone Survey Analysis

# ForeverAlone was a subreddit to share the forever alone meme, but somewhere down the line, it turned into an identity and a place where people who have been alone most of their lives could come and talk about their issues.
# 
# Tag line of r/ForeverAlone [A subreddit for Forever Alone. lonely depressed sad anxiety](https://www.reddit.com/r/ForeverAlone/)

# ## Import Data

# In[ ]:


# import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as plty
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load dataset
df = pd.read_csv('../input/foreveralone.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.duplicated().any()


# In[ ]:


df.isna().any()


# In[ ]:


df.job_title.value_counts()


# ## Data Wrangling

# <ul>
#     <li> Change Datatype of friends from float to int, because it's not possible to have 5.5 friends.</li>
#     <li> Remove rows where job_title is null. </li>
#     <li> Rename job_title values where the values is either in lower case or has a space before or after.  </li>
# </ul>

# Convert Datatype of Friends from float to int

# In[ ]:


# change dataype to int
df['friends'] = df['friends'].astype(np.int64)


# In[ ]:


df.info()


# Drop rows where it's null.

# In[ ]:


#drop rows with null values
df.dropna(inplace=True)


# In[ ]:


df.isna().any()


# Rename Values of Job_titles

# In[ ]:


# strip stings with white space
df['job_title'] = df.job_title.str.strip()


# In[ ]:


# Function to replace job_title values
def replace_text(what, to):
    df.replace(what, to, inplace= True)


# In[ ]:


replace_text('student', 'Student')
replace_text('none', 'None')
replace_text("N/a", 'None')
replace_text('na', 'None')
replace_text('-', 'None')
replace_text('.', 'None')
replace_text('*', 'None')
replace_text('ggg', 'None')


# In[ ]:


df.job_title.value_counts()


# ## EDA

# In[ ]:


df.gender.value_counts()


# In[ ]:


# Gender counts

data = [go.Bar(x = ['Male', 'Female', 'Transgender Male', 'Transgender Female'],
              y = df.gender.value_counts())]
layout = go.Layout(
    title='Gender Frequency',
    xaxis=dict(
        title='Gender'
    ),
    yaxis=dict(
        title='Count'
        )
    )

fig = go.Figure(data=data, layout=layout)
plty.offline.iplot(fig)


# In[ ]:


# sexuality freqency

df.sexuallity.value_counts()


# In[ ]:


# Sexuality counts

data = [go.Bar(x = ['Straight', 'Bisexual', 'Gay/Lesbian'],
              y = df.sexuallity.value_counts())]
layout = go.Layout(
    title='Sexuality Frequency',
    xaxis=dict(
        title='Sexuality'
    ),
    yaxis=dict(
        title='Count'
        )
    )

fig = go.Figure(data=data, layout=layout)
plty.offline.iplot(fig)


# In[ ]:


# body weight

df.bodyweight.value_counts()


# In[ ]:


def univariate_bar(column, ttitle, xlabel, ylabel):
    temp = pd.DataFrame({column:df[column].value_counts()})
    df1 = temp[temp.index != 'Unspecified']
    df1 = df1.sort_values(by=column, ascending=False)
    data  = go.Data([
                go.Bar(
                  x = df1.index,
                  y = df1[column],
            )])
    layout = go.Layout(
            title = ttitle,
        xaxis=dict(
            title=xlabel
        ),
        yaxis=dict(
            title=ylabel
            )
    )
    fig  = go.Figure(data=data, layout=layout)
    return plty.offline.iplot(fig)


# In[ ]:


univariate_bar('bodyweight', 'Bodyweight Frequency', 'Weight', 'Counts')


# In[ ]:


univariate_bar('depressed', 'Number of People Depressed', ' ', 'Count')


# In[ ]:


univariate_bar('social_fear', 'Number of People having Social Fear', ' ', 'Count')


# In[ ]:


univariate_bar('attempt_suicide', 'Number of people attempted suicide', ' ', 'Count')


# In[ ]:


age = df['age']

trace = go.Histogram(x = age)

data = [trace]

layout = go.Layout(
    title = 'Age Distribution',
    xaxis = dict(
        title = 'Age'
    ),
    yaxis = dict(
        title ='Count'
    ))

fig = go.Figure(data, layout)
plty.offline.iplot(fig)


# In[ ]:


# Distribution of Friends

friends = df['friends']

trace = go.Histogram(x = friends)
data = [trace]

layout = go.Layout(
    title = 'Friends Distribution',
    xaxis = dict(
    title = 'Friend Count'),
    yaxis = dict(
    title = 'Count')
    )

fig = go.Figure(data, layout)
plty.offline.iplot(fig)


# In[ ]:


male = df[df['gender'] == 'Male' ]
female = df[df['gender'] == 'Female' ]

male_age = male['age']
female_age = female['age']
trace1 = go.Histogram(x = male_age, 
                      name = 'Male',
                     opacity = 0.5)
trace2 = go.Histogram(x = female_age,
                      name = 'Female',
                     opacity = 0.5)

data = [trace1, trace2]

layout = go.Layout(
    title = 'Age Distribution on Gender',
    barmode='overlay',
    xaxis = dict(
    title = 'Age'),
    yaxis = dict(
    title = 'Count')
    )

fig = go.Figure(data, layout)
plty.offline.iplot(fig)


# In[ ]:


male_friends = male['friends']
female_friends = female['friends']
trace1 = go.Histogram(x = male_friends, 
                      name = 'Male',
                     opacity = 0.5)
trace2 = go.Histogram(x = female_friends,
                      name = 'Female',
                     opacity = 0.5)

data = [trace1, trace2]

layout = go.Layout(
    title = 'Friends Distribution on Gender',
    barmode='overlay',
    xaxis = dict(
    title = 'Friends'),
    yaxis = dict(
    title = 'Count')
    )

fig = go.Figure(data, layout)
plty.offline.iplot(fig)


# ### Conclusion
# *  Most of the people are in thier mid-age i.e between 18-30.
# *  Most of them have friends between 0-9.
# *  Almost more than half of them haven't attempted suicide.
# *  Most of them are either depressed or have social fear.
# 
