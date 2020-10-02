#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://images.unsplash.com/photo-1549224061-4c4818747ff9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60',width=400,height=400)


# Image Unsplash by Luemen Carlson

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df = pd.read_csv('../input/cusersmarildownloadsabortioncsv/abortion.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)
df.dataframeName = 'abortion.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


categorical_cols = [cname for cname in df.columns if
                    df[cname].nunique() < 10 and 
                    df[cname].dtype == "object"]


# Select numerical columns
numerical_cols = [cname for cname in df.columns if 
                df[cname].dtype in ['int64', 'float64']]


# In[ ]:


print(categorical_cols)


# In[ ]:


print(numerical_cols)


# Missing values.  Codes from my friend Caesar Lupum @caesarlupum

# In[ ]:


total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(8)


# In[ ]:


# Number of each type of column
df.dtypes.value_counts()


# In[ ]:


corrs = df.corr()
corrs


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


plt.figure(figsize = (20, 8))

# Heatmap of correlations
sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# EDA

# In[ ]:


sns.distplot(df["abortion_rate"])


# In[ ]:


sns.scatterplot(x='abortion_rate',y='period',data=df)


# In[ ]:


sns.countplot(df["age_of_woman"])


# In[ ]:


sns.countplot(df["abortion_rate"])


# A doughnut (or donut) chart serves a similar purpose to a pie chart, except that it is able to show more than one set of data.

# In[ ]:


import plotly.offline as pyo
import plotly.graph_objs as go
lowerdf = df.groupby('abortion_rate').size()/df['period'].count()*100
labels = lowerdf.index
values = lowerdf.values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
fig.show()


# In[ ]:


labels1=df.abortion_rate.value_counts().index
sizes1=df.abortion_rate.value_counts().values
plt.figure(figsize=(11,11))
plt.pie(sizes1,labels=labels1,autopct="%1.1f%%")
plt.title("abortion_rate",size=25)
plt.show()


# In[ ]:


labels1=df.age_of_woman.value_counts().index
sizes1=df.age_of_woman.value_counts().values
plt.figure(figsize=(11,11))
plt.pie(sizes1,labels=labels1,autopct="%1.1f%%")
plt.title("age_of_woman",size=25)
plt.show()


#  (crushed)Boxplots. Codes by Fatih Bilgin.

# In[ ]:


from plotly.offline import iplot


# In[ ]:


trace1 = go.Box(
    y=df["abortion_rate"],
    name = 'abortion_rate',
    marker = dict(color = 'rgb(0,145,119)')
)
trace2 = go.Box(
    y=df["period"],
    name = 'period',
    marker = dict(color = 'rgb(255, 111, 145)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='abortion_rate', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))
fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


ax = sns.swarmplot(x="abortion_rate", y="period", data=df)


# In[ ]:


print ("Skew is:", df.abortion_rate.skew())
plt.hist(df.abortion_rate, color='pink')
plt.show()


# In[ ]:


target = np.log(df.abortion_rate)
print ("Skew is:", target.skew())
plt.hist(target, color='green')
plt.show()

