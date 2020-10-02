#!/usr/bin/env python
# coding: utf-8

# <center><h1>2019 Data Science Bowl</h1></center>
# <center><h3>Uncover the factors to help measure how young children learn</h3></center>
# ![](https://cdn.pixabay.com/photo/2013/07/12/12/15/child-145411_1280.png)

# Time for Data Science Bowl 2019! In the fifth Data Science Bowl, we have the opportunity to look into early childhood education, and the role of digital media in unlocking a child's potentials! A very interesting topic, as a lot of parents I know have a pre-conception that their child's future is rather being ruined by mobile phones and digital media. 
# 
# In this competition, we have a data of kids playing a game app with five main challenges or assessments. Before each challenge, there are fun activities, games, and video clips. The attempts made to complete the challenge is the label in our dataset. Our job is to build a model to predict how many attempts it will take for a child to complete the challenge. 

# In[ ]:


# our imports
import numpy as np
import pandas as pd
import plotly.express as px


# In[ ]:


# loading all the data into dataframe
specs_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
train_labels_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
train_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')


# ## The Datasets

# First, we have three main datasets, train/test, train label, and specs. Train/test has the event data. Specs has more information (metadata) for each event-type. Let's print all of them for a quick idea. 
# 
# As you can see, Train/test and Specs both have the field event_id, and the dataframe can be merged with to each other using this field. There are 386 types of events, and metadata is available for every one of them in specs.csv file. 

# In[ ]:


specs_df.head()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# As the name suggests, train_labels.csv has the label (accuracy_group) per game session. As explained in the data section of this competition, there are four accuracy groups, also listed below for convenience. 
# 
# * 3: the assessment was solved on the first attempt
# * 2: the assessment was solved on the second attempt
# * 1: the assessment was solved after 3 or more attempts
# * 0: the assessment was never solved

# In[ ]:


train_labels_df.head()


# ## Installation ID
# 
# In the train data, we have the installation_id, which represents each application installed in a device. This can almost be treated as one to one mapping per child, but of course we have to take this field with a grain of salt (according to the competition data description). Let's plot the number of event counts per installation_id and see how noisy this field is.

# In[ ]:


# count activity per installation_id
count_per_installation_id = train_df.groupby(['installation_id']).count()['event_id']
# use plotly express to draw a scatterplot of activity per installation_id
fig = px.scatter(x=count_per_installation_id.index, y=count_per_installation_id.values,
                title='Total Events Per Installation')
fig.show()


# That's pretty noisy! We also have a field called game_session, which is a randomly generated unique identifier grouping events within a single game or video play session. Let's use this field to see how many sessions are there per installation.

# In[ ]:


session_per_installation_id = train_df.groupby(['installation_id']).game_session.nunique()
fig = px.scatter(x=session_per_installation_id.index, y=session_per_installation_id.values,
                title='Sessions Per Installation')
fig.show()


# Almost the same noise pattern follows for total event count and session per installation id. As you can see, a lot of installation ID have way too many game sessions or events to be considered as used by a single child!

# ![](https://cdn.pixabay.com/photo/2012/04/01/18/55/work-in-progress-24027_1280.png)
