#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# * In this kernel, we will learn how to use plotly library.
#     * Plotly library: The plotly Python library (plotly.py) is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.
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
# 
# 1. [Inset Plots](#11)
# 1. [3D Scatter Plot with Colorscaling](#12)
# 1. [Multiple Subplots](#13)
# 
# 
# Source: [DATAI Team](https://www.kaggle.com/kanncaa1)

# In[ ]:


#pip install plotly


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id="1"></a> <br>
# # Loading Data and Explanation of Features
# <font color='red'>
# * timesData includes 14 features that are:
#     <font color='black'>
#     * world_rank             
#     * university_name       
#     * country               
#     * teaching                
#     * international            
#     * research                 
#     * citations                
#     * income                   
#     * total_score              
#     * num_students             
#     * student_staff_ratio      
#     * international_students   
#     * female_male_ratio        
#     * year 

# In[ ]:


# Load data that we will use.
timesData = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")


# In[ ]:


timesData.shape


# In[ ]:


timesData.head()


# In[ ]:


timesData.info()


# In[ ]:


timesData.isna().sum()


# In[ ]:


timesData.student_staff_ratio.value_counts()


# In[ ]:


timesData.info()


# <a id="2"></a> <br>
# # Line Charts
# <font color='red'>
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
df = timesData.iloc[:100,:]

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
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
# # Scatter
# <font color='red'>
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


df2014 = timesData[timesData.year == 2014].iloc[:100,:]
df2015 = timesData[timesData.year == 2015].iloc[:100,:]
df2016 = timesData[timesData.year == 2016].iloc[:100,:]


# In[ ]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df2014['world_rank'], y=df2014['citations'],
                    mode='markers',
                    name='2014',
                    text= df2014.university_name))
fig.add_trace(go.Scatter(x=df2015['world_rank'], y=df2015['citations'],
                    mode='markers',
                    name='2015',
                    text= df2015.university_name))
fig.add_trace(go.Scatter(x=df2016['world_rank'], y=df2016['citations'],
                    mode='markers',
                    name='2016',
                    text= df2016.university_name))
# Add title
fig.update_layout(
    title="Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years",
    xaxis_title="World Rank",
    yaxis_title="Citation")


fig.show()


# <a id="4"></a> <br>
# # Bar Charts
# <font color='red'>
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


df2014 = timesData[timesData.year == 2014].iloc[:3,:]
df2014


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Citations', x=df2014.university_name, y=df2014.citations,text = df2014.country),
    go.Bar(name='Teaching', x=df2014.university_name, y=df2014.teaching,text = df2014.country)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# <font color='red'>
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


fig = go.Figure(data=[
    go.Bar(name='Citations', x=df2014.university_name, y=df2014.citations),
    go.Bar(name='Teaching', x=df2014.university_name, y=df2014.teaching)
])
# Change the bar mode
fig.update_layout(barmode='stack')

# Change the title
fig.update_layout(
    title="Citations and teaching of top 3 universities in 2014",
    xaxis_title="Top 3 universities")

fig.show()


# <a id="5"></a> <br>
# # Pie Charts
# <font color='red'>
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


df2016 = timesData[timesData.year == 2016].iloc[:7,:]
df2016


# In[ ]:


labels = df2016.university_name
values = [float(i.replace(',','.')) for i in df2016.num_students]


# In[ ]:


# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()


# <font color='red'>
# Second Pie Charts Example: Students rate of top 7 universities in 2016 (style2)
# 

# In[ ]:


# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
fig.show()


# <a id="6"></a> <br>
# # Bubble Charts
# <font color='red'>
# Bubble Charts Example: University world rank (first 20) vs teaching score with number of students(size) and international score (color) in 2016
# <font color='black'>
# * x = x axis
# * y = y axis
# * mode = markers(scatter)
# *  marker = marker properties
#     * color = third dimension of plot. Internaltional score
#     * size = fourth dimension of plot. Number of students
# * text: university names

# In[ ]:


df2016 = timesData[timesData.year == 2016].iloc[:20,:]
df2016


# In[ ]:


df2016.info()


# In[ ]:


# data preparation
df2016['world_rank'] = df2016['world_rank'].astype(float)
df2016['teaching'] = df2016['teaching'].astype(float)
df2016['international'] = df2016['international'].astype(float)
df2016['num_students'] = [float(i.replace(',','.')) for i in df2016.num_students]


# In[ ]:


fig = go.Figure(data=[go.Scatter(
    x=df2016['world_rank'],
    y=df2016['teaching'],
    text=df2016.university_name,
    mode='markers',
    marker=dict(
        color=df2016['international'],
        size=df2016['num_students'],
        showscale=True))])

fig.show()


# <a id="7"></a> <br>
# # Histogram
# <font color='red'>
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


# data preparation
df2011 = timesData[timesData.year == 2011]
df2012 = timesData[timesData.year == 2012]


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=df2011.student_staff_ratio,name='2011'))
fig.add_trace(go.Histogram(x=df2012.student_staff_ratio,name='2012'))

# Overlay both histograms
fig.update_layout(barmode='overlay')

fig.update_layout(
    title_text='Students-staff ratio in 2011 and 2012', # title of plot
    xaxis_title_text='Student-staff ratio', # xaxis label
    yaxis_title_text='Count') # yaxis label
    
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# <a id="8"></a> <br>
# # Word Cloud
# Not a pyplot but learning it is good for visualization. Lets look at which country is mentioned most in 2011.
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
                         ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# > <a id="9"></a> <br>
# # Box Plots
# <font color='red'>
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


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Box(y=x2015.total_score, name='Total score of universities in 2015',
                marker_color = 'indianred'))
fig.add_trace(go.Box(y=x2015.research, name = 'Research of universities in 2015',
                marker_color = 'lightseagreen'))

fig.show()


# <a id="10"></a> <br>
# # Scatter Matrix Plots
# <font color='red'>
# Scatter Matrix = it helps us to see covariance and relation between more than 2 features
# <font color='black'>
# * import figure factory as ff
# * create_scatterplotmatrix = creates scatter plot
#     * data2015 = prepared data. It includes research, international and total scores with index from 1 to 401
#     * colormap = color map of scatter plot
#     * colormap_type = color type of scatter plot
#     * height and weight

# In[ ]:


# prepare data
dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:,["research","international", "total_score"]]
data2015.index = np.arange(1, len(data2015)+1)
data2015["index"] = np.arange(1,len(data2015)+1)


# In[ ]:


data2015


# In[ ]:


# import figure factory
import plotly.figure_factory as ff

# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# <a id="11"></a> <br>
# # Inset Plots
# <font color='red'>
# Inset Matrix = 2 plots are in one frame
# <font color='black'>

# In[ ]:


import plotly as py
import plotly.graph_objs as go

trace1 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.teaching,
    name = "teaching")

trace2 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.income,
    xaxis='x2',
    yaxis='y2',
    name = "income")

data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2'),    
    title = 'Income and Teaching vs World Rank of Universities')

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="12"></a> <br>
# # 3D Scatter Plot with Colorscaling
# <font color='red'>
# 3D Scatter: Sometimes 2D is not enough to understand data. Therefore adding one more dimension increase the intelligibility of the data. Even we will add color that is actually 4th dimension.
# <font color='black'>
# * go.Scatter3d: create 3d scatter plot
# * x,y,z: axis of plots
# * mode: market that is scatter
# * size: marker size
# * color: axis of colorscale
# * colorscale:  actually it is 4th dimension

# In[ ]:


dataframe


# In[ ]:


import plotly.express as px

fig = px.scatter_3d(dataframe, x=dataframe.world_rank,
                        y=dataframe.research,
                        z=dataframe.citations)
fig.show()


#  <a id="13"></a> <br>
# # Multiple Subplots
# <font color='red'>
# Multiple Subplots: While comparing more than one features, multiple subplots can be useful.
# <font color='black'>
# 
# 

# In[ ]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize figure with subplots
fig = make_subplots(rows=2, cols=2, start_cell="bottom-left")

# Initialize figure with subplots
fig = make_subplots(
    rows=2, cols=2, subplot_titles=("research", "citations", "income", "total_score")
)


# Add traces
fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.research,name='research'), row=1, col=1)
fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.citations,name='citations'), row=1, col=2)
fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.income,name='income'), row=2, col=1)
fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.total_score,name='total_score'), row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="research", row=1, col=1)
fig.update_yaxes(title_text="citations", row=1, col=2)
fig.update_yaxes(title_text="income", row=2, col=1)
fig.update_yaxes(title_text="total_score", row=2, col=2)

# Update title and height
fig.update_layout(title_text="Research, citation, income and total score VS World Rank of Universities", height=700)


fig.show()


# In[ ]:




