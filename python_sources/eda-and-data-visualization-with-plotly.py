#!/usr/bin/env python
# coding: utf-8

# ### **<h1>INTRODUCTION</h1>**
# In this kernel,we will learn visualization with the plotly library.
# * **Plotly Library**: The Plotly library gives us an interactive visualization.
# * [Loading Data and Explanation of Features](#1)
# * Data Visualization
#    * [Line Chart](#2)
#    * [Scatter Plots](#3)
#    * [Bar Plot](#4)  
#    * [Pie Plot](#5) 
#    * [Bubble Charts](#6)
#    * [Histogram](#7)  
#    * [Word Cloud](#8)
#    * [Box Plot](#9)
#    * [Scatter Matrix Plots](#10)
#    * [Inset Plots](#11)
#    * [3D Scatter Plot](#12)
#    * [Multiple Subplots](#13) 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

#plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

#matplotlib library
import matplotlib.pyplot as plt

#word cloud library
from wordcloud import WordCloud

#from a unix time to a date
from time import strftime
from datetime import datetime

import warnings            
warnings.filterwarnings("ignore") 

import os
print(os.listdir("../input"))


# <a id=1></a>
# **<h1>Loading Data and Explanation of Features</h1>**
# ted_main contains 17 features that are:
# * **comments** : The number of first level comments made on the talk
# * **description** : A blurb of what the talk is about
# * **duration** : The duration of the talk in seconds
# * **event **: The TED/TEDx event where the talk took place
# * **film_date** : The Unix timestamp of the filming
# *  **languages **: The number of languages in which the talk is available
# * **main_speaker** : The first named speaker of the talk
# * **name** : The official name of the TED Talk. Includes the title and the speaker.
# * **num_speaker** : The number of speakers in the talk
# * **published_date** : The Unix timestamp for the publication of the talk on TED.com
# * **ratings** : A stringified dictionary of the various ratings given to the talk (inspiring, fascinating, jaw dropping, etc.)
# * **related_talks** : A list of dictionaries of recommended talks to watch next
# * **speaker_occupation** : The occupation of the main speaker
# * **tags** : The themes associated with the talk
# * **title** : The title of the talk
# * **url** : The URL of the talk
# * **views** : The number of views on the talk

# In[ ]:


#Load data from csv file
dataframe=pd.read_csv('../input/ted_main.csv')


# In[ ]:


#Let's get general information about our data
dataframe.info()


# It appears missing data in the speaker occupation feature. Let's visualize it. 
# * We can visually see the efficiency that is missing data with the missingno library.  Also we understand that it is not missing too much.

# In[ ]:


#rare visualization tool
#import missing library
import missingno as msno
msno.matrix(dataframe)
plt.show()


# let's continue and look at the first 5 lines of our data.

# In[ ]:


dataframe.head()


# <a id=0></a>
# **<h1>Line Chart</h1>**
# * It shows comments that speakers take.

# In[ ]:


trace1 = go.Scatter(
                    x = dataframe.index,
                    y = dataframe.comments,
                    mode = "lines",
                    name = "comments",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= dataframe.main_speaker)
data2 = [trace1]
layout = dict(title = 'Comment numbers for Ted Talks',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False)
             )
fig = dict(data = data2, layout = layout)
iplot(fig)


# The highest number of comments belong to Richard Dawkins. Let's learn a little more about him.

# In[ ]:


df=dataframe[dataframe['main_speaker']=='Richard Dawkins']
df


# According to the information we have, he participated in ted talks 3 times. 
# * Now let's try to find out the number of people who repeat in our dataset by joining different events at once.
#       The number below is the number of main speakers we have unique. Then we can say;
#       We have 2550 rows in our dataset, so 394 main_speaker feature has repeat data.

# In[ ]:


dataframe['main_speaker'].nunique()


# <a id=3></a>
# **<h1>Scatter Plot</h1>**
# * Let's look at the numbers of people who spoke at TEDGlobal 2005, TED2002, Royal Institution.

# In[ ]:


dfGlobal=dataframe[dataframe['event']=='TEDGlobal 2005']
df2002=dataframe[dataframe['event']=='TED2002']
dfRoyal=dataframe[dataframe['event']=='Royal Institution']

trace1 =go.Scatter(
                    x = dfGlobal.index,
                    y = dfGlobal.views,
                    mode = "markers",
                    name = "TEDGlobal 2005",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= dfGlobal.main_speaker)
# creating trace2
trace2 =go.Scatter(
                    x = df2002.index,
                    y = df2002.views,
                    mode = "markers",
                    name = "TED2002",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2002.main_speaker)
# creating trace3
trace3 =go.Scatter(
                    x = dfRoyal.index,
                    y = dfRoyal.views,
                    mode = "markers",
                    name = "Royal Institution",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= dfRoyal.main_speaker)
data3 = [trace1, trace2, trace3]
layout = dict(title = 'Number of views received at TEDGlobal 2005, TED2002, Royal Institution',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Views',ticklen= 5,zeroline= False)
             )
fig = dict(data = data3, layout = layout)
iplot(fig)


# <a id=4></a>
# **<h1>Bar Charts</h1>**

# * The number of comments and conversation times for speakers were visualized according to the publication date of the 
# speeches.
# * A bar plot was used for 6 speakers with the highest number of comments.

# In[ ]:


#sort by highest number of comments
data_sorted=dataframe.sort_values(by='comments',ascending=False)
#convert unix timestamp
data_sorted['published_date']=[datetime.fromtimestamp(int(item)).strftime('%Y') for item in data_sorted.published_date]
#get 6 speakers with the highest number of comments
data_comments=data_sorted.iloc[:6,:]
#duration convert  to minute
import datetime
data_duration=[]
data_duration=[str(datetime.timedelta(seconds=i))+" minute " for i in data_comments.duration]
date=[]
for item in data_comments.published_date:
    date.append(item + 'Year')

#visualization
#create trace1
trace1 = go.Bar(
                x = date,
                y = data_comments.comments,
                name = "comments",
                marker = dict(color = 'rgba(255, 58, 255, 0.4)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_comments.main_speaker)
# create trace2 
trace2 = go.Bar(
                x = date,
                y = data_comments.duration,
                name = "duration",
                marker = dict(color = 'rgba(15, 15, 250, 0.4)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = (data_duration + data_comments.main_speaker))
data4 = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data=data4, layout=layout)
iplot(fig)


# * The number of views and talk times of 3 people who received the most number of views.

# In[ ]:


#get 3 speakers with the highest number of comments
data_comments=data_sorted.iloc[:3,:]
#visualization
trace1 = {
  'x': data_comments.main_speaker,
  'y': data_comments.comments,
  'name': 'comments',
  'type': 'bar',
  'marker':dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
   'opacity':0.6,
};
trace2 = {
  'x': data_comments.main_speaker,
  'y': data_comments.duration,
  'name': 'duration',
  'type': 'bar',
  'text':data_duration,
  'marker':dict(
        color='rgb(158,202,225)',
        line=dict(color='rgb(8,48,107)',
                    width=1.5)),
  'opacity':0.6,
};
data5 = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 3 speakers'},
  'barmode': 'relative',
  'title': 'Number of comments and speech duration of the 3 most commented'
};
fig = go.Figure(data = data5, layout = layout)
iplot(fig)


# <a id=5></a>
# **<h1>Pie Plot</h1>**
# *  Views rate of events published in 2006  

# In[ ]:


#from a unix time to a date
from time import strftime
from datetime import datetime

dataframe['published_date']=[datetime.fromtimestamp(int(item)).strftime('%Y') for item in dataframe.published_date]
data_2006=dataframe[dataframe.published_date=='2006'].iloc[:,:]
labels=data_2006.event
# figure
fig = {
  "data": [
    {
      "values": data_2006.views,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Views Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"The number of watched talks events published in 2006",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Views",
                "x": 0.30,
                "y": 1.10
            },
        ]
    }
}
iplot(fig)


#   * The most views rate belongs to the TED2006 event and the least watched rate belongs to the TEDSalon 2006 event. 

# <a id=6></a>
# **<h1>Bubble Charts</h1>**
# * With Bubble Charts, we can look at the comments of 20 speakers who have the highest number of views. Also if the number of comments received is too high, the size of the bubble charts will increase accordingly. Rich darkening depends on the speech duration of the speakers.

# In[ ]:


data_sorted=dataframe.sort_values(by='views',ascending=True)
df=data_sorted.iloc[:20,:]
df.index=range(0,len(df))
#visualization
data = [
    {
        'y': df.views,
        'x': df.index,
        'mode': 'markers',
        'marker': {
            'color': df.duration,
            'size': df.comments,
            'showscale': True
        },
        "text" :  df.main_speaker    
    }
]
iplot(data)


# <a id=7></a>
# **<h1>Histogram</h1>**
# * Have a look at the frequency of comments in 2014 and 2015 with histogram graph.

# In[ ]:


data_2014=dataframe.comments[dataframe.event=='TED2014']
data_2015=dataframe.comments[dataframe.event=='TED2015']
    
trace2 = go.Histogram(
    x=data_2014,
    opacity=0.75,
    name = "2014",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))
trace3 = go.Histogram(
    x=data_2015,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(125, 2, 100, 0.6)'))
data = [trace2, trace3]
layout = go.Layout(barmode='overlay',
                   title=' Comments in 2014 and 2015',
                   xaxis=dict(title='number of comments'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id=8></a>
# **<h1>Word Cloud</h1>**
# Word Cloud is not in the plotly library.
# * Visualization of the names of the tags shared in 2017. If the number of tags is more, the names of the tags are bigger.

# In[ ]:


data_2017=dataframe.tags[dataframe.published_date=='2017']
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(data_2017))
plt.imshow(wordcloud)
plt.axis('off')


plt.show()


# In[ ]:





# <a id=9></a>
# **<h1>Box Plot</h1>**
# * At the TED2012 event, we can look at the number of comments and talk times with Box Plot. Talk times are in seconds. So the maximum speaking duration is approximately 25 minutes (1501 seconds) and the minimum speaking duration is approximately 4 minutes (181 seconds). Also we can see the outlines with the box plot in the comments.

# In[ ]:


data_2012=dataframe[dataframe.event=='TED2012']
#visualization
trace0 = go.Box(
    y=data_2012.comments,
    name = 'number of comments in TED2012',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=data_2012.duration,
    name = 'number of duration in TED2012',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)


# * Let's learn about the person with the least conversation time.

# In[ ]:


data_2012[data_2012.duration==181]


# <a id=10></a>
# **<h1>Scatter Matrix Plots</h1>**
# * We can look at the correlation between the number of comments and the number of views received by speakers entrepreneurs.

# In[ ]:


# import figure factory
import plotly.figure_factory as ff
df_occupation=dataframe[dataframe.event=='TED2012']
data_occupation = df_occupation.loc[:,["comments", "views"]]
data_occupation['index'] = np.arange(1,len(data_occupation)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data_occupation, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# <a id=11></a>
# **<h1>Inset Plots</h1>**
# * We can look at both the number of views and the number of comments for the first 100 data in our data.

# In[ ]:


#duration convert  to minute
import datetime
data_duration2=[]
data_duration2=[str(datetime.timedelta(seconds=i))+" minute " for i in dataframe.duration]
df_100=dataframe.iloc[:100,:]
#visualization
# first line plot
trace1 = go.Scatter(
    x=df_100.index,
    y=df_100.views,
    name = "views",
    marker = dict(color = 'rgba(200, 75, 45, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=df_100.index,
    y=df_100.duration,
    xaxis='x2',
    yaxis='y2',
    name = "duration",
    text=data_duration2,
    marker = dict(color = 'rgba(85, 20, 200, 0.8)'),
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Views and Comments'
)
fig = go.Figure(data=data, layout=layout)
plt.savefig('graph.png')
iplot(fig)
plt.show()


# <a id=12></a>
# **<h1>3D Scatter Plot</h1>**
# * We can look at the number of comments and the talk times related to the maximum number of views in 3D.

# In[ ]:



data_sorted2=dataframe.sort_values(by='views',ascending=False)
df_150=data_sorted2.iloc[:150,:]
df_150['views_rank']=np.arange(1,len(df_150)+1)

x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 400).transpose()
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=df_150.views_rank,
    y=df_150.comments,
    z=df_150.duration,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    ),
    text=data_duration2,)
data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id=13></a>
# **<h1>Multiple Plots</h1>**
# * After the 3D scatter plot, we can look at the number of comments and the duration of the speech with multiple subplots.

# In[ ]:


trace1 = go.Scatter(
    x=df_150.views_rank,
    y=df_150.comments,
    xaxis='x3',
    yaxis='y3',
    name = "comments"
)
trace2 = go.Scatter(
    x=df_150.views_rank,
    y=df_150.duration,
    xaxis='x4',
    yaxis='y4',
    name = "duration",
    text=data_duration2,
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Number of Comments and Number of Duration VS Number of Views Rank '
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

