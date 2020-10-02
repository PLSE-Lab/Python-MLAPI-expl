#!/usr/bin/env python
# coding: utf-8

# # Exploring, Visualizing And Preprocessing The Santander Dataset
# Exploring and visualizing the distributions and train-test differences of the Santander dataset is the ain focus of this notebook.
# 
# Have a good day!
# 
# # Importing Libraries

# In[1]:


# To store data
import pandas as pd

# To do linear algebra
import numpy as np

# To create plots
import matplotlib.pyplot as plt

# To create interactive plots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)

# To count things
from collections import Counter


# # Loading Data

# In[2]:


# Loading data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Create target and id
target = train.pop('target')
id_train = train.pop('ID')
id_test = test.pop('ID')

print('Train Shape: {}'.format(train.shape))
print('Test Shape: {}'.format(test.shape))


# In[3]:


train.head()


# # Target

# In[4]:


title = 'Histogram: Target, Log(Target) And Log10(Target) Santander Dataset'

fig = tools.make_subplots(rows=3, cols=1)

data_1 = go.Histogram(x = target, # y for rotated graph
                    histnorm = 'count', #'probability'
                    name = 'Target',
                    marker = dict(color = '#1b9e77'),
                    opacity = 1.0,
                    cumulative = dict(enabled = False))

data_2 = go.Histogram(x = np.log(target), # y for rotated graph
                    histnorm = 'count', #'probability'
                    name = 'Log(Target)',
                    marker = dict(color = '#d95f02'),
                    opacity = 1.0,
                    cumulative = dict(enabled = False))

data_3 = go.Histogram(x = np.log10(target), # y for rotated graph
                    histnorm = 'count', #'probability'
                    name = 'Log10(Target)',
                    marker = dict(color = '#7570b3'),
                    opacity = 1.0,
                    cumulative = dict(enabled = False))

fig.append_trace(data_1, 1, 1)
fig.append_trace(data_2, 2, 1)
fig.append_trace(data_3, 3, 1)

layout = go.Layout(title = title,
                   bargap = 0.2,
                   bargroupgap = 0.1)
fig['layout'].update(title=title, bargap=0.2)
fig['layout']['xaxis1'].update(title='Target')
fig['layout']['xaxis2'].update(title='Log(Target)')
fig['layout']['xaxis3'].update(title='Log10(Target)')
fig['layout']['yaxis1'].update(title='Count')
fig['layout']['yaxis2'].update(title='Count')
fig['layout']['yaxis3'].update(title='Count')

iplot(fig)


# Regarding the wide range of targets, predicting the log or log10 of the target will be advisable.
# 
# # Dataset

# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


print('Missing Values Train:\t{}'.format(train.isna().sum().sum()))
print('Missing Values Test:\t{}'.format(test.isna().sum().sum()))


# In[9]:


# Eliminate columns with a single value
single_value_columns = [col for col in train.columns if len(train[col].unique())==1]
train.drop(single_value_columns, axis=1, inplace=True)
test.drop(single_value_columns, axis=1, inplace=True)
print('Columns with single value in train: {}'.format(len(single_value_columns)))

print('Train Shape: {}'.format(train.shape))
print('Test Shape: {}'.format(test.shape))


# In[10]:


# Computing min for each column
column_mins = [train[col].min() for col in train.columns]
Counter(column_mins)


# The minimum for each column is 0.

# In[11]:


# Computing min for each column
column_maxs = [train[col].max() for col in train.columns]
min(column_maxs)


# Every column has at least a range from 0 to 2000 in its values. Computing the log or log10 for each column seems advisable.

# # Preprocessed Training Data

# In[ ]:


train = np.log(train+1)
test = np.log(test+1)


# In[ ]:




