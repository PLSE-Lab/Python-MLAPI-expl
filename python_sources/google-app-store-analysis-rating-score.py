#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
py.init_notebook_mode(connected=True)
import seaborn as sns

from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio

import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/googleplaystore.csv")
df1 = pd.read_csv("../input/googleplaystore_user_reviews.csv")
df = df.dropna()


# In[ ]:


df.head()


# In[ ]:


df1.dropna(inplace=True)


# In[ ]:


df1.head()


# In[ ]:


### Show Average Sentiment polarity of each Apps based on their customer Reviews

for x in df1.App.unique():
    (df1[df1.App==x].Sentiment_Polarity.mean())


# In[ ]:


df.Reviews = [int(x) for x in df.Reviews]
df.Rating = [float(x) for x in df.Rating]
df.Installs = [int(str(x).replace(",","").replace("+","")) for x in df.Installs]


# In[ ]:


### Showing Scatter Plot : number of Reviews vs Rating Scores

x = [int(x) for x in df.Reviews]
trace1 = go.Scatter(    x= [float(x) for x in df.Reviews],
                        y= [x for x in df['Rating']],
                        name='Review vs Rating',
                        mode='markers',
                        marker = dict(
                        size = 5,
                        color = 'rgba(255, 182, 193, .9)',
                        opacity=0.6,
                        line = dict(
                        width = 1,
                        )
                       )
            )
data = [trace1]

layout = go.Layout(
        xaxis=dict(
            range=[0, max(x)/2]
        ),
        yaxis=dict(
            range=[0, 6]
        ),
        title="Reviews vs Ratings"
    )
fig = go.Figure(data=data,layout=layout)

iplot(fig)


# In[ ]:


g = sns.jointplot(np.log(df.Reviews),df.Rating,kind='hex',height=10,ratio=5)
plt.show()


# In[ ]:


sns.jointplot(np.log(df.Reviews),df.Rating, kind='reg',ylim=(0,6),height=10,
              joint_kws={'line_kws':{'color':'red'}})


# In[ ]:


### making function across_column which is getting Column name as input and shows cross bar-plot
def across_column(column):
    ### Making Dictionary recording Average Review Scoring across App Genres

    tmp = dict()
    tmp = tmp.fromkeys(df[column].unique())

    for x in df[column].unique():
        tmp[x] = df[df[column]==x].Rating.mean()

    tmp = sorted(tmp.items(),key=lambda x:x[1],reverse=True)

    ### Average Reviews scoring across their Genres

    plt.figure(figsize=(10,25))
    x = [x[1] for x in tmp]
    y = [x[0] for x in tmp]
    sns.barplot(y=y,x=x,orient='h')


# In[ ]:


across_column('Genres')


# In[ ]:


df.head()


# In[ ]:


across_column('Price')


# In[ ]:


df.head()


# In[ ]:


free_R=df[df.Type=='Free'].Rating.mean()
paid_R=df[df.Type!='Free'].Rating.mean()

sns.boxplot(x=['Free','Paid'],y=[free_R,paid_R])


# In[ ]:


pd.get_dummies(df[['Category','Type','Content Rating','Genres']])

