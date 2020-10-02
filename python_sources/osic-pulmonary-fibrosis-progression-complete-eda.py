#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd 
import numpy as np

import pydicom

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import seaborn as sns
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')


# # Train Data EDA

# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.isnull().sum()


# So we have no null values.

# In[ ]:


def uniques(df):
    print("UNIQUE VALUE STATS")
    print(f"{len(df)} Rows")
    print("Column\t\tUniquevalues")
    for col in df.columns:
        print(f"{col}\t\t{len(df[col].unique())}")


# In[ ]:


uniques(train_df)


# ## Exploring Age Field

# In[ ]:


fig = px.histogram(train_df, x="Age")
fig.update_layout(title_text='Age Distribution')
fig.show()


# ## Exploring Weeks Field

# In[ ]:


fig = px.histogram(train_df, x="Weeks",marginal="rug")
fig.update_layout(title_text='Weeks Distribution')
fig.show()


# ## Exploring the Sex Field

# In[ ]:


fig = px.histogram(train_df, x="Sex")
fig.update_layout(title_text='Sex Counts')
fig.show()


# 3x male data compared to Female

# ## Exploring SmokingStatus Field

# In[ ]:


fig = px.histogram(train_df, x="SmokingStatus")
fig.update_layout(title_text='Smoking Status')
fig.show()


# ## Exploring Relationships

# In[ ]:


fig = px.scatter(train_df, x="FVC", y="Percent")
fig.update_layout(title_text='Percent vs FVC')
fig.show()


# Somewhat of a linear distribution here between percent and FVC. Makes sense as both terms are proportional.

# In[ ]:


fig = px.scatter(train_df, x="FVC", y="Age" , color ="Sex")
fig.update_layout(title_text='Age vs FVC in terms of Sex')
fig.show()


# Males have Higher FVC irrespective of Age

# In[ ]:


fig = px.scatter(train_df, x="Weeks", y="Percent" , color ="Sex")
fig.update_layout(title_text='Percent vs Weeks')
fig.show()


# So the percentage doesnt show any specific trend with the weeks passed.

# In[ ]:


fig = px.scatter(train_df, x="Weeks", y="FVC" , color ="Sex")
fig.update_layout(title_text='FVC vs Weeks')
fig.show()


# No specific trends

# In[ ]:


fig = px.bar(train_df, y='FVC', x='SmokingStatus')
fig.update_layout(title = 'FVC based of Smoking Status')
fig.show()


# People with History of smoking clearly show higher FVC levels <br>
# Values cannot be compared with each other as number of datapoints vary ( ie. data is unbalanced )

# In[ ]:


fig = px.histogram(train_df, x="SmokingStatus", color='Sex')
fig.update_layout(title_text='Smoking Status')
fig.show()


# In[ ]:


fig = px.histogram(train_df, x="Age", color='Sex')
fig.update_layout(title_text='Smoking Status')
fig.show()


# # Exploring Test Data

# In[ ]:


test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


def getHisto(col):
    fig = px.histogram(test_df, x=col)
    fig.update_layout(title_text=col + ' Distribution')
    fig.show()


# In[ ]:


getHisto("Weeks")


# In[ ]:


getHisto("Age")


# In[ ]:


getHisto("Sex")


# In[ ]:


getHisto("SmokingStatus")


# # Exploring Image Data

# In[ ]:


filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"
ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 


# In[ ]:


train_df.loc[train_df.Patient == 'ID00007637202177411956430']


# In[ ]:


os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')


# In[ ]:




