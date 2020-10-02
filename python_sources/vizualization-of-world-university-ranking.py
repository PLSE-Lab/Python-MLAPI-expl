#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


timesData = pd.read_csv('../input/timesData.csv')


# In[ ]:


timesData.info()


# In[ ]:


timesData.head(10)


#  **LINE CHARTS**
# 
# Import graph_objs as go
# Creating traces
# x = x axis
# y = y axis
# mode = type of plot like marker, line or line + markers
# name = name of the plots
# marker = marker is used with dictionary.
# color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
# text = The hover text (hover is curser)
# data = is a list that we add traces into it
# layout = it is dictionary.
# title = title of layout
# x axis = it is dictionary
# title = label of x axis
# ticklen = length of x axis ticks
# zeroline = showing zero line or not
# fig = it includes data and layout
# iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


df = timesData.iloc[:100, :]

import plotly.graph_objs as go

# Trace1
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text = df.university_name)

# Trace2
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text = df.university_name)

data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching World Rank of Top100 Universities',
              xaxis = dict(title = 'World Rank', ticklen = 5, zeroline = False))
fig = dict(data = data, layout = layout)
iplot(fig)


# **SCATTER**
# 
# Import graph_objs as go
# Creating traces
# x = x axis
# y = y axis
# mode = type of plot like marker, line or line + markers
# name = name of the plots
# marker = marker is used with dictionary.
# color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
# text = The hover text (hover is curser)
# data = is a list that we add traces into it
# layout = it is dictionary.
# title = title of layout
# x axis = it is dictionary
# title = label of x axis
# ticklen = length of x axis ticks
# zeroline = showing zero line or not
# y axis = it is dictionary and same with x axis
# fig = it includes data and layout
# iplot() = plots the figure(fig) that is created by data and layout
# 

# In[ ]:


df2014 = timesData[timesData.year == 2014].iloc[:100, :]
df2015 = timesData[timesData.year == 2015].iloc[:100, :]
df2016 = timesData[timesData.year == 2016].iloc[:100, :]

import plotly.graph_objs as go

#trace1
trace1 = go.Scatter(
                    x = df2014.world_rank,
                    y = df2014.citations,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = df2014.university_name)
#trace2
trace2 = go.Scatter(
                    x = df2015.world_rank,
                    y = df2015.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = df2015.university_name)
#trace3
trace3 = go.Scatter(
                    x = df2016.world_rank,
                    y = df2016.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = df2016.university_name)

data = [trace1, trace2, trace3]
layout = dict(title = 'Citation and World Rank of Top100 Universities with 2014, 2015 and 2016 years',
              xaxis = dict(title = 'World Rank', ticklen = 5,zeroline = False),
              yaxis = dict(title = 'Citations', ticklen = 5, zeroline = False))
fig = dict(data = data, layout = layout)
iplot(fig)


# **BAR CHARTS**
# 
# Import graph_objs as go
# Creating traces
# x = x axis
# y = y axis
# mode = type of plot like marker, line or line + markers
# name = name of the plots
# marker = marker is used with dictionary.
# color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
# line = It is dictionary. line between bars
# color = line color around bars
# text = The hover text (hover is curser)
# data = is a list that we add traces into it
# layout = it is dictionary.
# barmode = bar mode of bars like grouped
# fig = it includes data and layout
# iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


df2014 = timesData[timesData.year == 2014].iloc[:3, :]
df2014


# In[ ]:


import plotly.graph_objs as go

# trace1
trace1 = go.Bar(
                x = df2014.university_name,
                y = df2014.citations,
                name = 'citations',
                marker = dict(color = 'rgba(255, 84, 255, 0.5)',
                             line = dict(color = 'rgb(0,0,0)', width = 1.5)),
                text = df2014.country)
# trace2
trace2 = go.Bar(
                x = df2014.university_name,
                y = df2014.teaching,
                name = 'teaching',
                marker = dict(color = 'rgba(255, 200, 128, 0.5)',
                              line = dict(color = 'rgb(0,0,0)', width = 1.5)),
                text = df2014.country)

data = [trace1, trace2]
layout = go.Layout(barmode = 'group')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


df2014 = timesData[timesData.year == 2014].iloc[:3,:]

import plotly.graph_objs as go

x = df2014.university_name

# trace1
trace1 = {
    'x' : x,
    'y' : df2014.citations,
    'name' : 'citation',
    'type' : 'bar'};

# trace2
trace2 = {
    'x' : x,
    'y' : df2014.teaching,
    'name' : 'teaching',
    'type' : 'bar'};

data = [trace1, trace2];
layout = {
    'xaxis' : {'title' : 'Top 3 Universities'},
    'barmode' : 'relative',
    'title' : 'Citations and Teaching of Top3 Universities in 2014'};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **PIE CHARTS**
# 
# fig: create figures
# data: plot type
# values: values of plot
# labels: labels of plot
# name: name of plots
# hoverinfo: information in hover
# hole: hole width
# type: plot type like pie
# layout: layout of plot
# title: title of layout
# annotations: font, showarrow, text, x, y

# In[ ]:


df2016 = timesData[timesData.year == 2016].iloc[:7, :]
pie = df2016.num_students
pie1_list = [float(each.replace(',','.')) for each in df2016.num_students]
labels = df2016.university_name

fig = {
    "data" : [
        {
            "values" : pie1_list,
            "labels" : labels,
            "domain" : {'x' : [0, 0.5]},
            "name" : 'Number of Students Rates',
            "hoverinfo" : 'label+percent+name',
            "hole" : 0.3,
            "type" : 'pie'
        },
    ],
    "layout" : {
        "title" : "Number of Students Rates at Universities",
        "annotations" : [
            {
                "font" : { "size" : 20},
                "showarrow" : False,
                "text" : "Number of Students",
                "x" : 0.05,
                "y" : 1,
            },
        ]
    }
}
iplot(fig)


# **BUBBLE CHARTS**
# 
# x = x axis
# y = y axis
# mode = markers(scatter)
# marker = marker properties
# color = third dimension of plot. Internaltional score
# size = fourth dimension of plot. Number of students
# text: university names

# In[ ]:


df2016


# In[ ]:


df2016 = timesData[timesData.year == 2016].iloc[:20, :]
num_students_size = [float(each.replace(',', '.')) for each in df2016.num_students]
international_color = [float(each) for each in df2016.international]
data = [
    {
        'x' : df2016.world_rank,
        'y' : df2016.teaching,
        'mode' : 'markers',
        'marker' : {
            'color' : international_color,
            'size' : num_students_size,
            'showscale' : True
        },
        'text' : df2016.university_name
    }
]
iplot(data)


# **HISTOGRAM**
# 
# trace1 = first histogram
# x = x axis
# y = y axis
# opacity = opacity of histogram
# name = name of legend
# marker = color of histogram
# trace2 = second histogram
# layout = layout
# barmode = mode of histogram like overlay. Also you can change it with stack

# In[ ]:


x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace1 = go.Histogram(
                    x = x2011,
                    opacity = 0.75,
                    name = '2011',
                    marker = dict(color = 'rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
                    x = x2012,
                    opacity = 0.75,
                    name = '2012',
                    marker = dict(color = 'rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]

layout = go.Layout(
                xaxis = dict(title = 'Students-Staff Ratio'),
                yaxis = dict(title = 'Count'),
                barmode = 'overlay',
                title = 'Student-Staff Ratio in 2011 and 2012')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **WORD CLOUD**
# 
# WordCloud = word cloud library that I import at the beginning of kernel
# background_color = color of back ground
# generate = generates the country name list(x2011) a word cloud

# In[ ]:


x2011 = timesData.country[timesData.year == 2011]
plt.subplots(figsize = (8,8))
wordcloud = WordCloud(
                    background_color = 'white',
                    width = 512,
                    height = 384
                                ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()


# **BOX PLOTS**
# 
# Median (50th percentile) = middle value of the data set. Sort and take the data in the middle. It is also called 50% percentile that is 50% of data are less that median(50th quartile)(quartile)
# 
# 25th percentile = quartile 1 (Q1) that is lower quartile
# 75th percentile = quartile 3 (Q3) that is higher quartile
# height of box = IQR = interquartile range = Q3-Q1
# Whiskers = 1.5 * IQR from the Q1 and Q3
# Outliers = being more than 1.5*IQR away from median commonly.
# 
# trace = box
# y = data we want to visualize with box plot
# marker = color

# In[ ]:


x2015 = timesData[timesData.year == 2015]

trace1 = go.Box(
                y = x2015.total_score,
                name = 'Total Score of Universities in 2015',
                marker = dict(color = 'rgb(12, 12, 140)'))
trace2 = go.Box(
                y = x2015.research,
                name = 'Research of Universities in 2015',
                marker = dict(color = 'rgb(12, 128, 128)'))

data = [trace1, trace2]
iplot(data)


# **SCATTER MATRIX PLOTS**
# 
# mport figure factory as ff
# create_scatterplotmatrix = creates scatter plot
# data2015 = prepared data. It includes research, international and total scores with index from 1 to 401
# colormap = color map of scatter plot
# colormap_type = color type of scatter plot
# height and weight

# In[ ]:


import plotly.figure_factory as ff

dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:, ["research", "international", "total_score"]]
data2015["index"] = np.arange(1, len(data2015)+1)

fig = ff.create_scatterplotmatrix(data2015, diag = 'box', index = 'index', colormap = 'Portland', 
                                  colormap_type = 'cat', width = 700, height = 700)
iplot(fig)


# **INSET PLOTS**

# In[ ]:


trace1 = go.Scatter(
                    x = dataframe.world_rank,
                    y = dataframe.teaching,
                    name = 'Teaching',
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
trace2 = go.Scatter(
                    x = dataframe.world_rank,
                    y = dataframe.income,
                    xaxis = 'x2',
                    yaxis = 'y2',
                    name = 'Income',
                    marker = dict(color = 'rgba(160, 112, 20, 0.8)'))
data = [trace1, trace2]
layout = go.Layout(
                    xaxis2 = dict(domain = [0.6, 0.99], anchor = 'y2'),
                    yaxis2 = dict(domain = [0.6, 0.95], anchor = 'x2'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **3D SCATTER PLOT with COLORSCALING**
# 
# go.Scatter3d: create 3d scatter plot
# x,y,z: axis of plots
# mode: market that is scatter
# size: marker size
# color: axis of colorscale
# colorscale: actually it is 4th dimension

# In[ ]:


trace1 = go.Scatter3d(
                    x = dataframe.world_rank,
                    y = dataframe.research,
                    z = dataframe.citations,
                    mode = 'markers',
                    marker = dict(size = 10, color = 'rgb(255,0,0)'))
data = [trace1]
layout = go.Layout(margin = dict(l = 0, r = 0, b = 0, t = 0))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **MULTIPLE SUBPLOTS**

# In[ ]:


trace1 = go.Scatter(
                    x = dataframe.world_rank,
                    y = dataframe.research,
                    name = 'Research')
trace2 = go.Scatter(x = dataframe.world_rank,
                    y = dataframe.citations,
                    xaxis = 'x2',
                    yaxis = 'y2',
                    name = 'Citations')
trace3 = go.Scatter(x = dataframe.world_rank,
                    y = dataframe.income,
                    xaxis = 'x3',
                    yaxis = 'y3',
                    name = 'Income')
trace4 = go.Scatter(x = dataframe.world_rank,
                    y = dataframe.total_score,
                    xaxis = 'x4',
                    yaxis = 'y4',
                    name = 'Total Score')
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
                xaxis = dict(domain = [0, 0.45]),
                yaxis = dict(domain = [0, 0.45]),
                xaxis2 = dict(domain = [0.55, 1]),
                yaxis2 = dict(domain = [0, 0.45], anchor = 'x3'),
                xaxis3 = dict(domain = [0, 0.45], anchor = 'y3'),
                yaxis3 = dict(domain = [0.55, 1]),
                xaxis4 = dict(domain = [0.55, 1], anchor = 'y4'),
                yaxis4 = dict(domain = [0.55, 1], anchor = 'x4'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# Thanks.
