#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fancyimpute import KNN
import pandas as pd
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[ ]:


df = pd.read_excel('../input/Absenteeism_at_work_Project.xls')
df.shape


# In[ ]:


df.head()


# In[ ]:


columns = df.columns


# In[ ]:


df['Body mass index'].head()


# In[ ]:


df['Body mass index'].isna().sum()


# In[ ]:


df['Body mass index']


# In[ ]:


df['Body mass index'].iloc[12]


# In[ ]:


df['Body mass index'].iloc[12] = np.nan


# In[ ]:


df['Body mass index']


# In[ ]:


np.asarray(df['Body mass index'].dropna(),dtype=float).mean()


# In[ ]:


mean = 26.69


# In[ ]:


from scipy import stats
stats.mode(np.asarray(df['Body mass index'].dropna(),dtype=float))


# In[ ]:


mode = 31


# In[ ]:


np.median(np.asarray(df['Body mass index'].dropna(),dtype=float))


# In[ ]:


median = 25


# In[ ]:


knn = KNN(k=3)
df = knn.fit_transform(df)


# In[ ]:


type(df)


# In[ ]:


df = pd.DataFrame(df,columns=columns)


# In[ ]:


type(df)


# In[ ]:


df['Body mass index'].isna().sum()


# In[ ]:


round(df['Body mass index'].iloc[12])


# In[ ]:


df['Body mass index']


# In[ ]:


trace0 = go.Box(
    y=df['Transportation expense']
)
data = [trace0]
py.iplot(data)


# In[ ]:


trace0 = go.Box(
    y = df['Height'],
    name = "All Points",
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = 'rgb(7,40,89)'),
    line = dict(
        color = 'rgb(7,40,89)')
)

trace1 = go.Box(
    y = df['Height'],
    name = "Only Whiskers",
    boxpoints = False,
    marker = dict(
        color = 'rgb(9,56,125)'),
    line = dict(
        color = 'rgb(9,56,125)')
)

trace2 = go.Box(
    y = df['Height'],
    name = "Suspected Outliers",
    boxpoints = 'suspectedoutliers',
    marker = dict(
        color = 'rgb(8,81,156)',
        outliercolor = 'rgba(219, 64, 82, 0.6)',
        line = dict(
            outliercolor = 'rgba(219, 64, 82, 0.6)',
            outlierwidth = 2)),
    line = dict(
        color = 'rgb(8,81,156)')
)

trace3 = go.Box(
    y = df['Height'],
    name = "Whiskers and Outliers",
    boxpoints = 'outliers',
    marker = dict(
        color = 'rgb(107,174,214)'),
    line = dict(
        color = 'rgb(107,174,214)')
)

data = [trace0,trace1,trace2,trace3]

layout = go.Layout(
    title = "Box Plot Styling Outliers"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Box Plot Styling Outliers")


# # Outlier analysis

# In[ ]:


def calculate_outliers(column):
    
    # Calculating 25th and 75th percentile.
    percentile25 = np.percentile(column,25)
    print("perctile 25",percentile25)
    percentile75 = np.percentile(column,75)
    print("percentile 75",percentile75)
    
    diff = percentile75 - percentile25
    
    lowerlimit = percentile25 - diff * 1.5
    upperlimit = percentile75 + diff * 1.5
    
    column[column<lowerlimit] = np.nan
    column[column>upperlimit] = np.nan
    
    return column


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.boxplot(df['Body mass index'])


# In[ ]:


columns


# In[ ]:


plt.boxplot(df['Height'])


# In[ ]:


plt.boxplot(df['Work load Average/day '])


# In[ ]:


plt.boxplot(df['Transportation expense'])


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


df['Transportation expense'] = calculate_outliers(df['Transportation expense'])


# In[ ]:


df['Transportation expense'].isna().sum()


# In[ ]:


knn = KNN(k=3)
df = knn.fit_transform(df)
df = pd.DataFrame(df,columns=columns)


# In[ ]:


df.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




