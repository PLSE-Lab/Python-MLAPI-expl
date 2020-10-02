#!/usr/bin/env python
# coding: utf-8

# # PLOTLY
# 
# <br>Content:
# 1. [Loading Data and Explanation of Features](#1)
# 1. [Line Charts](#2)
# 1. [Scatter Charts](#3)
# 1. [Bar Charts](#4)
# 1. [Pie Charts](#5)
# 1. [Bubble Charts](#6)
# 1. [Histogram](#7)
# 1. [Word Cloud](#8)
# 1. [Box Plot](#9)
# 1. [Scatter Plot Matrix](#10)
# 1. [Inset Plots](#11)
# 1. [3D Scatter Plot with Colorscaling](#12)
# 1. [Multiple Subplots](#13)
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> 
# ## 1-Loading Data and Explanation of Features

# In[ ]:


#loading data
timesData= pd.read_csv("../input/world-university-rankings/timesData.csv")


# In[ ]:


timesData.info()


# In[ ]:


timesData.head(15)


# <a id="2"></a> <br>
# # 2)Line Charts
# <font color='blue'>
# Line Charts Example: Citation and Teaching vs World Rank of Top 100 Universities
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * title = title of layout
#     * x axis = it is dictionary
#         * title = label of x axis
#         * ticklen = length of x axis ticks
#         * zeroline = showing zero line or not
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frame
df = timesData.iloc[:100,:]     #  ":100"  means take the first 100 sample   # ",:" means take all the features

# import graph objects as "go"
import plotly.graph_objs as go


# In[ ]:


# Creating trace1
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(110, 2, 3, 0.8)'),
                    text= df.university_name)
# Creating trace2
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)
data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id="3"></a> <br>
# # 3)Scatter
# <font color='blue'>
# Scatter Example: Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * title = title of layout
#     * x axis = it is dictionary
#         * title = label of x axis
#         * ticklen = length of x axis ticks
#         * zeroline = showing zero line or not
#     * y axis = it is dictionary and same with x axis
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


timesData['year'].unique()
#Here we can see the unique values of year feature


# In[ ]:


# prepare data frames
df2014 = timesData[timesData.year == 2014].iloc[:100,:]       #take the first 100 sample of the year 2014
df2015 = timesData[timesData.year == 2015].iloc[:100,:]       #take the first 100 sample of the year 2015
df2016 = timesData[timesData.year == 2016].iloc[:100,:]       #take the first 100 sample of the year 2016
# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2014.world_rank,
                    y = df2014.citations,
                    mode = "markers",   #marker=scatter
                    name = "2014",
                    marker = dict(color = 'rgba(0, 0, 255, 0.8)'),
                    text= df2014.university_name)
# creating trace2
trace2 =go.Scatter(
                    x = df2015.world_rank,
                    y = df2015.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= df2015.university_name)
# creating trace3
trace3 =go.Scatter(
                    x = df2016.world_rank,
                    y = df2016.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(200, 100, 200, 0.8)'),
                    text= df2016.university_name)
data = [trace1, trace2, trace3]
layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id="4"></a> <br>
# # 4)Bar Charts
# <font color='blue'>
# First Bar Charts Example: citations and teaching of top 3 universities in 2014 (style1)
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#         * line = It is dictionary. line between bars
#             * color = line color around bars
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * barmode = bar mode of bars like grouped
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frames
df2014 = timesData[timesData.year == 2014].iloc[:3,:]
df2014


# In[ ]:


# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = df2014.university_name,
                y = df2014.citations,
                name = "citations",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)    #text = pointer on the chart
# create trace2 
trace2 = go.Bar(
                x = df2014.university_name,
                y = df2014.teaching,
                name = "teaching",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")  #barmode=group citations and teaching side by side
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <font color='blue'>
# Second Bar Charts Example: citations and teaching of top 3 universities in 2014 (style2)
# <br> Actually, if you change only the barmode from *group* to *relative* in previous example, you achieve what we did here. However, for diversity I use different syntaxes. 
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * name = name of the plots
#     * type = type of plot like bar plot
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * xaxis = label of x axis
#     * barmode = bar mode of bars like grouped( previous example) or relative
#     * title = title of layout
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frames
df2014 = timesData[timesData.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go

x = df2014.university_name

trace1 = {
  'x': x,
  'y': df2014.citations,
  'name': 'citation',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': df2014.teaching,
  'name': 'teaching',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 3 universities'},
  'barmode': 'relative',
  'title': 'citations and teaching of top 3 universities in 2014'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="5"></a> <br>
# # 5)Pie Charts
# <font color='blue'>
# Pie Charts Example: Students rate of top 7 universities in 2016
# <font color='black'>
# * fig: create figures
#     * data: plot type
#         * values: values of plot
#         * labels: labels of plot
#         * name: name of plots
#         * hoverinfo: information in hover
#         * hole: hole width
#         * type: plot type like pie
#     * layout: layout of plot
#         * title: title of layout
#         * annotations: font, showarrow, text, x, y

# In[ ]:


# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = df2016.university_name
# figure
#actually we create this chart different way but it can also be created the way we did before.
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .2,
      "type": "pie"
    },],
  "layout": {
        "title":"Universities Number of Students rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Students",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)


# <a id="6"></a> <br>
# # 6)Bubble Charts
# <font color='blue'>
# Bubble Charts Example: University world rank (first 20) vs teaching score with number of students(size) and international score (color) in 2016
# <font color='red'>
#  We are gonna built 4 dimentional chart(x,y,size,colour)
# <font color='black'>
# * x = x axis
# * y = y axis
# * mode = markers(scatter)
# *  marker = marker properties
#     * color = third dimension of plot. Internaltional score
#     * size = fourth dimension of plot. Number of students
# * text: university names

# In[ ]:


# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:20,:]
num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]
international_color = [float(each) for each in df2016.international]
data = [
    {
        'y': df2016.teaching,
        'x': df2016.world_rank,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': num_students_size,
            'showscale': True
        },
        "text" :  df2016.university_name    
    }
]
iplot(data)


# <a id="7"></a> <br>
# # 7)Histogram
# <font color='blue'>
# Lets look at histogram of students-staff ratio in 2011 and 2012 years. 
#     <font color='black'>
# * trace1 = first histogram
#     * x = x axis
#     * y = y axis
#     * opacity = opacity of histogram
#     * name = name of legend
#     * marker = color of histogram
# * trace2 = second histogram
# * layout = layout 
#     * barmode = mode of histogram like overlay. Also you can change it with *stack*

# In[ ]:


# prepare data
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace1 = go.Histogram(
    x=x2011,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name = "2012",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="8"></a> <br>
# # 8)Word Cloud
# Not a pyplot but good for visualization. Lets look at which country is mentioned most in 2011.
# * WordCloud = word cloud library that I import at the beginning of kernel
#     * background_color = color of back ground
#     * generate = generates the country name list(x2011) a word cloud

# In[ ]:


# data prepararion
x2011 = timesData.country[timesData.year == 2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x2011))  #seperating the words
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# <a id="9"></a> <br>
# # 9)Box Plots
# <font color='blue'>
# * Box Plots
#     * Median (50th percentile) = middle value of the data set. Sort and take the data in the middle. It is also called 50% percentile that is 50% of data are less that median(50th quartile)(quartile)
#         * 25th percentile = quartile 1 (Q1) that is lower quartile
#         * 75th percentile = quartile 3 (Q3) that is higher quartile
#         * height of box = IQR = interquartile range = Q3-Q1
#         * Whiskers = 1.5 * IQR from the Q1 and Q3
#         * Outliers = being more than 1.5*IQR away from median commonly.
#         
#     <font color='black'>
#     * trace = box
#         * y = data we want to visualize with box plot 
#         * marker = color

# In[ ]:


# data preparation
x2015 = timesData[timesData.year == 2015]

trace0 = go.Box(
    y=x2015.total_score,
    name = 'total score of universities in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=x2015.research,
    name = 'research of universities in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)


# <a id="10"></a> <br>
# # 10)Scatter Matrix Plots
# <font color='blue'>
# Scatter Matrix = it helps us to see covariance and relation between more than 2 features
# <font color='black'>
# * import figure factory as ff
# * create_scatterplotmatrix = creates scatter plot
#     * data2015 = prepared data. It includes research, international and total scores with index from 1 to 401
#     * colormap = color map of scatter plot
#     * colormap_type = color type of scatter plot
#     * height and weight

# In[ ]:


# import figure factory
import plotly.figure_factory as ff
# prepare data
dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:,["research","international", "total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# <font color='red'>
# As you can see there is a positive correlation between research and total score, in total score and research there are outliers.

# <a id="11"></a> <br>
# # 11)Inset Plots
# <font color='blue'>
# Inset Matrix = 2 plots are in one frame
# <font color='black'>

# In[ ]:


# first line plot
trace1 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.teaching,
    name = "teaching",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.income,
    xaxis='x2',
    yaxis='y2',
    name = "income",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
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
    title = 'Income and Teaching vs World Rank of Universities'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="12"></a> <br>
# # 12) 3D Scatter Plot with Colorscaling
# <font color='blue'>
# 3D Scatter: Sometimes 2D is not enough to understand data. Therefore adding one more dimension increase the intelligibility of the data. Even we will add color that is actually 4th dimension.
# <font color='black'>
# * go.Scatter3d: create 3d scatter plot
# * x,y,z: axis of plots
# * mode: market that is scatter
# * size: marker size
# * color: axis of colorscale
# * colorscale:  actually it is 4th dimension

# In[ ]:


# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=dataframe.world_rank,
    y=dataframe.research,
    z=dataframe.citations,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',    # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0    #left, right, bottom, top margin(distance from edges)
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="13"></a> <br>
# # Multiple Subplots
# <font color='red'>
# Multiple Subplots: While comparing more than one features, multiple subplots can be useful.
# <font color='black'>
# 
# 

# In[ ]:


trace1 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.research,
    name = "research"
)
trace2 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.citations,
    xaxis='x2',
    yaxis='y2',
    name = "citations"
)
trace3 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.income,
    xaxis='x3',
    yaxis='y3',
    name = "income"
)
trace4 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.total_score,
    xaxis='x4',
    yaxis='y4',
    name = "total_score"
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Research, citation, income and total score VS World Rank of Universities'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

