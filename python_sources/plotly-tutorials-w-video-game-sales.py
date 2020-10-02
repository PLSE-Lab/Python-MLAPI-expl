#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


gamesdata = pd.read_csv("../input/vgsales.csv")


# In[ ]:


gamesdata.info()


# In[ ]:


gamesdata.columns


# In[ ]:


gamesdata.head()


# In[ ]:


gamesdata.tail()


# #
# **LINE CHARTS**
#  
#  *Line Charts Example: NA Sales vs EU Sales of Top 100 Games*
#  * Import graph_objs as go
#  * Creating traces
#      * x = x axis (Usually for parameters)
#      * y = y axis (Usually for values)   
#      * mode = type of plot like marker, line or line + markers
#      * name = name of the plots
#      * marker = marker is used with dictionary
#          * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#      * text = The hover text (hover is curser)
#  * data = is a list that we add traces into it
#  * layout = it is dictionary
#      * title = title of layout
#      * x axis = it is dictionary
#          * title = label of x axis
#          * ticklen = length of x axis ticks
#          * zeroline = showing zero line or not
#  * fig = it includes data and layout  
#  * iplot() = plots the figure(fig) that is created by data and layout
# 

# In[ ]:


#preparing data frame
df = gamesdata.iloc[:100,:]

#import graph objects as "go"
import plotly.graph_objs as go

#creating trace1
trace1 = go.Scatter(
                    x = df.Rank,
                    y = df.NA_Sales,
                    mode = "lines",
                    name = "NA Sales",
                    marker = dict(color="rgba(166,11,2,0.8)"),
                    text = df.Name)
#creating trace2
trace2 = go.Scatter(
                    x = df.Rank,
                    y = df.EU_Sales,
                    mode = "lines+markers",
                    name = "EU Sales",
                    marker = dict(color = "rgba(80,12,160,0.5)"),
                    text = df.Name)
data = [trace1,trace2]
layout = dict(title = "Global Sales of Top 100 Games",
                xaxis = dict(title="Rank",ticklen= 5, zeroline=False)
             )
fig = dict(data = data, layout = layout)
py.offline.iplot(fig)


# #
# **BAR CHARTS**
#  
#  *In this example we'll search the answer of total sales of each gaming platform *
#  
# * Creating traces
#      * x = x axis (In this example, platforms)
#      * y = y axis (Values)
#      * name = name of the plots
#      * marker = marker is used with dictionary
#          * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#          * line = It is dictionary. line between bars 
#              *color = line color around bar
#      * text = The hover text (hover is curser) (we didn't need to use this)
#      * data = is a list that we add traces into it    
#      * layout = it is dictionary  
#          * barmode = bar mode of bars like grouped (Optionally you can use this in different DataSets)
#      * fig = it includes data and layout
#      * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


gamesdata["Platform"].unique()


# In[ ]:


platformlist = list(gamesdata["Platform"].unique())
platformearns = []
for i in platformlist:
    x = gamesdata[gamesdata["Platform"] == i]
    sums = sum(x.Global_Sales)
    platformearns.append(sums)
    
data = pd.DataFrame({"platformlist": platformlist, "platformearns": platformearns})
new_index = (data["platformearns"].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

#create trace1
trace1 = go.Bar(
                x = sorted_data.platformlist,
                y = sorted_data.platformearns,
                name = "Global Sales of Platforms",
                marker = dict(color = "rgba(10,255,50,0.7)",line = dict(color="rgb(0,0,0)",width=1)))
data = [trace1]
layout = dict(title = "Global Sales of Gaming Platforms")
fig = go.Figure(data = data, layout = layout)
py.offline.iplot(fig)


# #
# **PIE CHART**
#  
#  *In this example we'll look over the market leader games of 2016 with different use of figure because we have only one trace*
#  
#  * fig: create figures
#  
#      * data: plot type
#          * values: values of plot
#          * labels: labels of plot
#          * name: name of plots
#          * hoverinfo: information in hover
#          * hole: hole width
#          * type: plot type like pie
#      * layout: layout of plot
#          * title: title of layout
#          * annotations: font, showarrow, text, x, y
#          
# 

# In[ ]:


# data preparetion
df2016 = gamesdata[gamesdata.Year == 2016].iloc[:10,:]
pie = df2016.Global_Sales
labels = df2016.Name
publisher = df2016.Publisher
# figure

fig = {
    "data": [
        {
            "values": pie,
            "labels": labels,
            "domain":{"x": [0,.5]},
            "name": "Sale Rate",
            "hoverinfo": "label+percent+name",
            "hole": .3,
            "type":"pie"
        },],
    "layout": {
        "title":"Top 10 Market Leaders In 2016",
        "annotations": [
            { "font": { "size": 15},
              "showarrow": True,
              "text": "Games",
                "x": 0.20,
                "y": 1
    },]}

}
py.offline.iplot(fig)


# #
# **BUBBLE CHARTS**
# 
# *Bubble Charts Example: Top 20 sale values of publishers with size and color*
# 
# * x = x axis (rank)
# * y = y axis (publishers total income from games)
# * mode = markers(scatter)
# * marker = marker properties
#     * color = NA sales (3rd dimension)
#     * size = number of games that publishers made (4th dimension)
# * text = name of publishers   

# In[ ]:


#preparing data
publisherlist = list(gamesdata["Publisher"].unique())
publisherearns = []
for i in publisherlist:
    y = gamesdata[gamesdata["Publisher"] == i]
    sums1 = sum(y.Global_Sales)
    publisherearns.append(sums1)
    
data1 = pd.DataFrame({"Publisher": publisherlist,"publisherearns":publisherearns})
new_index = (data1["publisherearns"].sort_values(ascending=False).index.values)
sorted_data = data1.reindex(new_index)

df = sorted_data.iloc[:20]
df["Rank"] = "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"
df["NA_Sales"] = gamesdata.NA_Sales
#visualization
data = [
    {
        "x":df.Rank,
        "y":df.publisherearns,
        "mode": "markers",
        "marker":{
            "color": df.NA_Sales,
            "size": gamesdata.Publisher.value_counts()/10,
            "showscale": True
        },
        "text": df.Publisher
    }
]
py.offline.iplot(data)


# #
# **HISTOGRAM**
# 
# *In this example we'll find number of genres in 2010 and 2016*
# 
# * trace1 = first histogram
#     * x = x axis
#     * y = y axis
#     * opacity = opacity of histogram
#     * name = name of legend
#     * marker = color of histogram
# * trace2 = second histogram
# * layout = layout
#     * barmode = mode of histogram like overlay. Also you can change it with stack
# 

# In[ ]:


x2016 = gamesdata.Genre[gamesdata.Year == 2016]
x2006 = gamesdata.Genre[gamesdata.Year == 2010]

trace1 = go.Histogram(
                        x = x2016,
                        opacity = 0.75,
                        name = "2016",
                        marker = dict(color="rgba(162,50,70,0.9)"))
trace2 = go.Histogram(
                        x = x2006,
                        opacity = 0.75,
                        name = "2010",
                        marker = dict(color="rgba(24,68,200,0.6)"))

data = [trace1,trace2]
layout = go.Layout(barmode = "overlay",
                    title = "Number of Genres in 2016 and 2010 ",
                  xaxis = dict(title="Genre"),
                  yaxis = dict(title = "Count"),
                  )
fig = go.Figure(data=data,layout=layout)
py.offline.iplot(fig)


# #
# **3D SCATTER PLOT WITH COLORSCALING**
# 
# **3D Scatter: Sometimes 2D is not enough to understand data. Therefore adding one more dimension increase the intelligibility of the data. Even we will add color that is actually 4th dimension.**
# 
# * go.Scatter3d: create 3d scatter plot
# * x,y,z: axis of respectively Rank sales in Japan and sales in Europe
# * mode: market that is scatter
# * size: marker size
# * color: axis of colorscale
# * colorscale: actually it is 4th dimension
# 

# In[ ]:


trace1 = go.Scatter3d(
    x=gamesdata.Rank.iloc[:100],
    y=gamesdata.JP_Sales,
    z=gamesdata.EU_Sales,
    mode="markers",
    marker=dict(
        size=10,
        color="rgb(216,34,78)",#set color to an array/list of desired values
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
    )
)
fig = go.Figure(data=data,layout=layout)
py.offline.iplot(fig)

