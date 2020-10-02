#!/usr/bin/env python
# coding: utf-8

# <h1>Exploratory Analysis of Electoral Data</h1>
# Based on my previous cleaned data, I am continuing my work towards exploratory analysis.
# 
# I have added data source from my <b>Election Data Wrangling </b>kernal

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

import os
print(os.listdir("../input/election-data-wrangling"))

# Any results you write to the current directory are saved as output.


# In[ ]:


NA= pd.read_csv("../input/election-data-wrangling/NA2002-18.csv")
Cand = pd.read_csv("../input/election-data-wrangling/Canditates2018.csv")


# In[ ]:


Cand.rename(columns={'Seat':'ConstituencyTitle'}, inplace=True)
Cand.columns
cy = Cand
cy['Year'] = "2018"
df = pd.concat([NA, cy])


# In[ ]:


print(cy.columns, cy.shape)


# In[ ]:





# <h3> Total Candidates Appeared in All Elections </h3>

# It's strange to know that Candidates appeared in 2013 were twice as comapre to 2008.

# In[ ]:


data = [go.Bar(
    x=df['Year'].unique(),
    y=df.groupby(['Year'])['CandidateName'].count(),
    textposition = 'auto',
    marker=dict(
        color=['rgba(204,204,204,1)', 'rgba(204,204,204,1)',
               'rgba(222,45,38,0.8)', 'rgba(204,204,204,1)'],
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
    )]
layout = {
  'xaxis': {'title': 'Year'},
  'yaxis': {'title': 'No. of Participants'},
  'barmode': 'relative',
  'title': 'Total Number of Candidates Appeared'
};
iplot({'data': data, 'layout': layout})


# <h2>Which Party has High Representatives? </h2>
# Lets have a look which Political Party have high representatives. 

# In[ ]:


#df = NA
x=df['Party'].unique()

trace0 = go.Bar(
    x=x,
    y=df[df['Year']==2002].groupby(['Party'])['CandidateName'].count().nlargest(5),
    name="2002",
    marker=dict(
        color='rgb(49,130,189)'
    )
)

trace1 = go.Bar(
    x=x,
    y=df[df['Year']==2008].groupby(['Party'])['CandidateName'].count().nlargest(5),
    name="2008",
    marker=dict(
        color='rgb(204,104,204)',
    )
)

trace2 = go.Bar(
    x=x,
    y=df[df['Year']==2013].groupby(['Party'])['CandidateName'].count().nlargest(5),
    name="2013",
    marker=dict(
        color='rgb(104,104,104)',
    )
)

data = [trace0, trace1, trace2]

layout = {
  'xaxis': {'title': 'Year'},
  'yaxis': {'title': 'No. of Participants'},
  'barmode': 'group',
  'title': 'Total Number of Candidates Appeared'
};
iplot({'data': data, 'layout': layout})


# So, PTI Candidates increased Exponentialy. 
# 
# Let's have a look where they gained party representatives in 2013
# 

# In[ ]:


df = NA[NA['Party']=="Pakistan Tehreek-e-Insaf"]
x=df['ConstituencyTitle'].unique()

trace0 = go.Bar(
    x=x,
    y=df[df['Year']==2002].groupby(['ConstituencyTitle'])['CandidateName'].count(),
    name="2002",
    marker=dict(
        color='rgb(49,130,189)'
    )
)

trace1 = go.Bar(
    x=x,
    y=df[df['Year']==2008].groupby(['ConstituencyTitle'])['CandidateName'].count(),
    name="2008",
    marker=dict(
        color='rgb(204,104,204)',
    )
)

trace2 = go.Bar(
    x=x,
    y=df[df['Year']==2013].groupby(['ConstituencyTitle'])['CandidateName'].count(),
    name="2013",
    marker=dict(
        color='rgb(104,104,104)',
    )
)

data = [trace0, trace1, trace2]
layout = {
  'xaxis': {'title': 'Constituecy'},
  'yaxis': {'title': 'No. of Participants'},
  'barmode': 'group',
  'title': 'PTI Candidates over Constituency'
};

iplot({'data': data, 'layout': layout})


# To know PTI representation at District level.

# In[ ]:


df = NA[NA['Party']=="Pakistan Tehreek-e-Insaf"]
x=df['District'].unique()

trace0 = go.Bar(
    x=x,
    y=df[df['Year']==2002].groupby(['District'])['CandidateName'].count(),
    name="2002",
    marker=dict(
        color='rgb(49,130,189)'
    )
)

trace1 = go.Bar(
    x=x,
    y=df[df['Year']==2008].groupby(['District'])['CandidateName'].count(),
    name="2008",
    marker=dict(
        color='rgb(204,104,204)',
    )
)

trace2 = go.Bar(
    x=x,
    y=df[df['Year']==2013].groupby(['District'])['CandidateName'].count(),
    name="2013",
    marker=dict(
        color='rgb(104,104,104)',
    )
)

data = [trace0, trace1, trace2]
layout = {
  'xaxis': {'title': 'District'},
  'yaxis': {'title': 'No. of Participants'},
  'barmode': 'stack',
  'title': 'PTI Candidates over Constituency'
};

iplot({'data': data, 'layout': layout})


# The above plot gives a useful insight that PTI has not good representation of her Party in 2008. We can see their single representative in Peshawar in 2008. After 2008, they go wild and gained maximum representation in almost at areas.
# 
# <b>Lets have a look which Constituency has high worth where competition is high between parties. </b> 

# In[ ]:


df = NA
x=df['ConstituencyTitle'].unique()

trace0 = go.Bar(
    x=x,
    y=df[df['Year']==2002].groupby(['ConstituencyTitle'])['CandidateName'].count().nlargest(10),
    name="2002",
    marker=dict(
        color='rgb(49,130,189)'
    )
)

trace1 = go.Bar(
    x=x,
    y=df[df['Year']==2008].groupby(['ConstituencyTitle'])['CandidateName'].count().nlargest(10),
    name="2008",
    marker=dict(
        color='rgb(204,104,204)',
    )
)

trace2 = go.Bar(
    x=x,
    y=df[df['Year']==2013].groupby(['ConstituencyTitle'])['CandidateName'].count().nlargest(10),
    name="2013",
    marker=dict(
        color='rgb(104,104,104)',
    )
)

data = [trace0, trace1, trace2]
layout = go.Layout(
    xaxis=dict(tickangle=-45),
    barmode='group',
)

iplot(data)


# It reflects that these are very old constituency.

# <h2> Contour Graph </h2>
# A contour plot is a graph that you can use to explore the potential relationship between three variables.
# Contour plots display the 3-dimensional relationship in two dimensions, with x- and y-factors (predictors) plotted on the x- and y-scales and response values represented by contours.

# In[ ]:


d = NA[NA['Year']== 2013]
iplot([go.Histogram2dContour(x=d['Party'], 
                             y=d['Votes'], 
                             contours=go.Contours(coloring='heatmap'))])


# In[ ]:


d = NA[NA['Year']== 2013]
iplot([go.Histogram2dContour(x=d['District'], 
                             y=d['Votes'], 
                             contours=go.Contours(coloring='heatmap'))])


# In[ ]:


d = NA[NA['Year']== 2013]
iplot([go.Histogram2dContour(x=d['District'], 
                             y=d['Votes'], 
                             contours=go.Contours(coloring='heatmap')),
             go.Scatter(x=d['District'], y=d['Votes'], mode='markers')])


# In[ ]:


d = NA[NA['Year']== 2002]
iplot([go.Histogram2dContour(x=d['ConstituencyTitle'], 
                             y=d['Votes'].nlargest(10), 
                             contours=go.Contours(coloring='heatmap'))])


# <h2>Party Representatives Appearing in 2018 Elections </h2>

# In[ ]:


Cand.groupby(['PartyAcro'])['CandidateName'].count().nlargest(10)


# In[ ]:


x=Cand['PartyAcro'].unique()
data = [go.Bar(
    x=x,
    y=Cand.groupby(['PartyAcro'])['CandidateName'].count().nlargest(10),
    textposition = 'auto',
    marker=dict(
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
    )]
layout = {
  'xaxis': {'title': 'Party'},
  'yaxis': {'title': 'No. of Participants'},
  'barmode': 'relative',
  'title': 'Number of Candidates Appearing in 2018 Elections'
};
iplot({'data': data, 'layout': layout})


# In[ ]:


ax = Cand.groupby(['PartyAcro'])['CandidateName'].count().nlargest(10) .plot(kind='bar',
                                    figsize=(14,8),
                                    title="Total Number of Candidates Appearing in 2018")
ax.set_xlabel("Year")
ax.set_ylabel("No. of Candidates Participating in 2018")
plt.show()


# <b>I am tired .. Will continue later.
# 
# Stay Tuned! <b/>

# In[ ]:




