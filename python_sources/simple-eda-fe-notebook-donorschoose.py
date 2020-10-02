#!/usr/bin/env python
# coding: utf-8

# **Objective of this Notebook:**
# 
# In this notebook, let us try to do some Interactive Exploratory Data Analysis. Then we can concentrate on the Feature Engineering part followed by Modeling. The language in the notebook will be Python.
# 
# **About DonorsChoose:**
# 
# [DonorsChoose.org](https://www.donorschoose.org/) empowers public school teachers from across the country to request much-needed materials and experiences for their students. At any given time, there are thousands of classroom requests that can be brought to life with a gift of any amount.
# 
# **Objective of the competition:**
# 
# DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.The goal of the competition is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school. DonorsChoose.org can then use this information to identify projects most likely to need further review before approval. 
# 
# Let us start with importing the necessary modules.

# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[7]:


### Let us read the train file and look at the top few rows ###
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()


# In[8]:


train_df.shape


# We have around 182K rows in the dataset with 16 columns. Let us first look into the distribution of the target variable "project_is_approved" to understand more about the class imbalance.

# In[9]:


temp_series = train_df["project_is_approved"].value_counts()

labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Project Proposal is Approved'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="ProjectApproval")


# Nice to see that ~85% of the project proposals are approved. So we do have a class imbalance with the majority class as positive. It is good that we have [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) as evaluation metric, given this class imbalance. 
# 
# Now that we got an idea about the distribution of the classes, let us now look at the inidividual variables to understand more. 
# 
# **Project Grade Category:**

# In[44]:


### Stacked Bar Chart ###
x_values = train_df["project_grade_category"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["project_grade_category"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["project_grade_category"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["project_grade_category"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Grade Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance percentage by Project grade",
    width = 1000,
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# Project proposal acceptance percentage is 83 to 85% for all the Grades of the class. So may be this variable is not really important for our prediction. Now let us look at the next variable "project_subject_categories".
# 
# **Project Subject Category:**

# In[46]:


### Stacked Bar Chart ###
x_values = train_df["project_subject_categories"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["project_subject_categories"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["project_subject_categories"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["project_subject_categories"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Subject Category Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance percentage by Project Subject Category",
    yaxis=dict(range=[0.6, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# Project subject category has a very long tail. Among the top categories, the acceptance percentage lies from 80 to 86% and so slightly better than the previous feature. 
# 
# **Teacher Prefix:**

# In[47]:


### Stacked Bar Chart ###
x_values = train_df["teacher_prefix"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["teacher_prefix"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["teacher_prefix"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["teacher_prefix"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Teacher Prefix Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Teacher Prefix",
    width = 1000,
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# Among the prefixes, "Teacher" has low (79%) acceptance rate compared to others. Now let us look at the states at which the schools are located.
# 
# **School States:**

# In[49]:


### Stacked Bar Chart ###
x_values = train_df["school_state"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["school_state"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["school_state"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["school_state"]==val]))
    
trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "School State Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by School state",
    yaxis=dict(range=[0.75, 0.9])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# Now let us look at the same statewise distribution in US map for better visual understanding. 

# In[60]:


con_df = pd.DataFrame(train_df["school_state"].value_counts()).reset_index()
con_df.columns = ['state_code', 'num_proposals']

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = con_df['state_code'],
        z = con_df['num_proposals'].astype(float),
        locationmode = 'USA-states',
        text = con_df['state_code'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Num Project Proposals")
        ) ]

layout = dict(
        title = 'Project Proposals by US States<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


### mean acceptance rate ###
con_df = pd.DataFrame(train_df.groupby("school_state")["project_is_approved"].apply(np.mean)).reset_index()
con_df.columns = ['state_code', 'mean_proposals']

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = con_df['state_code'],
        z = con_df['mean_proposals'].astype(float),
        locationmode = 'USA-states',
        text = con_df['state_code'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Project Proposals Acceptance Rate")
        ) ]

layout = dict(
        title = 'Project Proposals Acceptance Rate by US States<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# Now let us look at the data from temporal point of view. 
# 
# **Project Submission Time:**
# 
# This is the time at which the application is submitted. 

# In[58]:


train_df["project_submitted_datetime"] = pd.to_datetime(train_df["project_submitted_datetime"])
train_df["date_created"] = train_df["project_submitted_datetime"].dt.date

x_values = train_df["date_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["date_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["date_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["date_created"]==val]))

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Date Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission date",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# Some of the inferences from the plots are:
# 
# * Looks like we have approximately one years' worth of data (May 2016 to April 2017) given in the training set. 
# * There is a sudden spike on a single day (Sep 1, 2016) with respect to the number of proposals (may be some specific reason?)
# * There is a spike in the number of proposals coming in during the initial part of the academic year (August till October)
# * There is no visible pattern as such in the acceptance rate based on the timeline. 
# 
# Now let us also look at the plots at different levels of temporal attributes like month, day of month, weekday etc.

# In[61]:


train_df["month_created"] = train_df["project_submitted_datetime"].dt.month

x_values = train_df["month_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["month_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["month_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["month_created"]==val]))

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Month Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission Month",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# September month has the second highest number of proposals and the least acceptance rate of all the months. 
# 
# 

# In[63]:


train_df["weekday_created"] = train_df["project_submitted_datetime"].dt.weekday

x_values = train_df["weekday_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["weekday_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["weekday_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["weekday_created"]==val]))
x_values = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Weekday Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission Weekday",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# The number of proposals decreases as we move towards the end of the week. Now let us look at the hour of submission pattern.

# In[64]:


train_df["hour_created"] = train_df["project_submitted_datetime"].dt.hour

x_values = train_df["hour_created"].value_counts().index.tolist()
y0_values = []
y1_values = []
y_values = []
for val in x_values:
    y1_values.append(np.sum(train_df["project_is_approved"][train_df["hour_created"]==val] == 1))
    y0_values.append(np.sum(train_df["project_is_approved"][train_df["hour_created"]==val] == 0))
    y_values.append(np.mean(train_df["project_is_approved"][train_df["hour_created"]==val]))

trace1 = go.Bar(
    x = x_values,
    y = y1_values,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = x_values,
    y = y0_values, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Hour Distribution",
    barmode='stack',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradeCategory')

### Bar chart ###
trace = go.Bar(
    x = x_values,
    y = y_values,
    name='Accepted Proposals'
)
data = [trace]
layout = go.Layout(
    title = "Project acceptance rate by Proposal Submission Hour",
    yaxis=dict(range=[0.7, 0.95])
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ProjectGradePerc')


# Hours 03 to 06 has the least number of proposals and acceptance rate is also marginally on the lower side. Now let us combine the resource dataset to get some more insights.

# In[ ]:


## Reading the data ##
resource_df = pd.read_csv("../input/resources.csv")

## Merging with train and test data ##
train_df = pd.merge(train_df, resource_df, on="id", how='left')
test_df = pd.merge(test_df, resource_df, on="id", how='left')

resource_df.head()


# **Things to do - Upcoming versions:**
# 1. Combine resources data and do exploratory analysis.
# 2. Build a basic model using these variables
# 3. Exploration of text based features
# 4. Build a model using both numeric and text based features
# 5. Creation of new features and rebuild the model. 

# **More to come. Stay tuned.!**
