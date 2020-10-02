#!/usr/bin/env python
# coding: utf-8

# BEFORE start up I would like to thank for[ DATAI Team](http://www.kaggle.com/kanncaa1) for their helps.
# * If you are a beginner and very excited to learn something new like me ;) here my kernels for you.
# * [Data ScienceTutorial for Beginners](http://www.kaggle.com/aliylmaz0907/data-sciencetutorial-for-beginners)
# * [SeaBorn](http://www.kaggle.com/aliylmaz0907/suicide-statistics-with-seaborn-library-1)
# * [SeaBorn2](http://www.kaggle.com/aliylmaz0907/homicide-reports-with-seaborn-library-2)
# 

# # INTRODUCTION
# * In this kernel, I will show how to use plotly library.
#     * Plotly library: Plotly's Python graphing library makes interactive, publication-quality graphs online. Examples of how to make line plots, scatter plots, area charts, bar charts, error bars, box plots, histograms, heatmaps, subplots, multiple-axes, polar charts, and bubble charts etc... But I will use only a few of them because of my data.
#     * [Plotly Library](http://plot.ly/python/) is Here for you
# 
# <br>Content:
# 1. [Loading Data and Explanation of Features](#1)
# 1. [Line Charts](#2)
# 1. [Scatter Charts](#3)
# 1. [Bar Charts](#4)
# 1. [Histogram](#5)
# 1. [Scatter Plot Matrix](#6)
# 1. [Word Cloud](#7)
# 1. [Box Plot](#8)
# 1. [Inset Plots](#9)
# 1. [3D Scatter Plot with Colorscaling](#10)
# 1. [Multiple Subplots](#11)
# 

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
# matplotlib
import matplotlib.pyplot as plt

# word cloud library
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> <br>
# # Loading Data and Explanation of Features
# <font color='red'>
# * Humans Freedoms Data  includes several features but we will be use few of them. Here some of examples:
#     <font color='black'>
#     * Rule of law             
#     * Region       
#     * Countries              
#     * Human Freedom (rank)             
#     * Human Freedom (score)            
#     * Economic Freedom (rank)                 
#     * Hours regulations for labour             
#     * Inflation: most recent year                  
#     * Government consumption         
#     * year 

# In[ ]:


humfree= pd.read_csv('../input/hfi_cc_2018.csv')


# In[ ]:


humfree.head()


# In[ ]:


humfree.info()


# # Before Starting Visualization Please use your cursor on the grafics.You will realize that these graphics are talking to us :)

# <a id="1"></a> <br>
# # Line Charts
# <font color='red'>
# Line Charts Example: Human Freedom Rank and Human Freedom Score of Countries
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
# 

# In[ ]:


#preparing Data
df2016 = humfree[humfree.year == 2016].iloc[:,:]
df2015 = humfree[humfree.year == 2015].iloc[:,:]
df2014 = humfree[humfree.year == 2014].iloc[:,:]
new_index = (df2016['hf_rank'].sort_values(ascending=True)).index.values
df2016 = df2016.reindex(new_index) # with this code we sort our data according to human freedom rank
# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2014.hf_rank,
                    y = df2014.hf_score,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2014.countries)
# creating trace2
trace2 =go.Scatter(
                    x = df2015.hf_rank,
                    y = df2015.hf_score,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2015.countries)
# creating trace3
trace3 =go.Scatter(
                    x = df2016.hf_rank,
                    y = df2016.hf_score,
                    mode = "lines",
                    name = "2016",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2016.countries)
data = [trace1, trace2, trace3]
layout = dict(title = 'Human Freedom score vs Human Freedom rank of Countries ',
              xaxis= dict(title= 'Freedom Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Freedom Score',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


pd.set_option("display.max_columns",123) # with this code we can see all the columns


# <a id="3"></a> <br>
# # Scatter
# <font color='red'>
# Scatter Example: Human Freedom Ranks and Rule of law Of each Countries  with 2014, 2015 and 2016 years
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


#preparing data
df2016 = humfree[humfree.year == 2016].iloc[:,:]
df2015 = humfree[humfree.year == 2015].iloc[:,:]
df2014 = humfree[humfree.year == 2014].iloc[:,:]
new_index = (df2016['hf_rank'].sort_values(ascending=True)).index.values
df2016 = df2016.reindex(new_index)
# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2014.hf_rank,
                    y = df2014.pf_rol,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(55, 157, 94, 0.8)'),
                    text= df2014.countries)
# creating trace2
trace2 =go.Scatter(
                    x = df2015.hf_rank,
                    y = df2015.pf_rol,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 18, 03, 0.8)'),
                    text= df2015.countries)
# creating trace3
trace3 =go.Scatter(
                    x = df2016.hf_rank,
                    y = df2016.pf_rol,
                    mode = "markers", # as you realised Scatter and Line charts are almost the same plot
                    name = "2016",     # we change only 'MODE' it is only 3 possibility 'marker, line or line + markers'
                    marker = dict(color = 'rgba(230, 25, 200, 0.8)'),
                    text= df2016.countries)
data = [trace1, trace2, trace3]
layout = dict(title = 'Rule of law vs Human Freedom rank of Countries ',
              xaxis= dict(title= 'Freedom Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Freedom Score',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id="4"></a> <br>
# # Bar Charts
# <font color='red'>
# First Bar Charts Example: Government enterprises and investments and Government consumption of 10 countries in 2016 (style1)
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
df2016 = humfree[humfree.year == 2016].iloc[:10,:]


# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = df2016.countries,
                y = df2016.ef_government_consumption,
                name = "Government consumption",
                marker = dict(color = 'rgba(55, 114, 55, 0.8)',# It takes RGB "0-255" for all values for opacity "0-1"
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2016.countries)
# create trace2 
trace2 = go.Bar(
                x = df2016.countries,
                y = df2016.ef_government_enterprises,
                name = "Government enterprises and investments",
                marker = dict(color = 'rgba(235, 155, 12, 0.9)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2016.countries)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <font color='red'>
# Second Bar Charts Example: Religion Harassment and physical hostilities and  Religion Legal and regulatory restrictions (style2)
# <br> Actually, if you change only the barmode from *group* to *relative* in previous example, you will see what I do here. However, for diversity I use different syntaxes. 
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
df2008 = humfree[humfree.year == 2008].iloc[:15,:]
# import graph objects as "go"
import plotly.graph_objs as go

x = df2008.countries

trace1 = {
  'x': x,
  'y': df2008.pf_religion_harassment,
  'name': 'Harassment and physical hostilities',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': df2008.pf_religion_restrictions,
  'name': 'Legal and regulatory restrictions',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': ' Countries at 2008'},
  'barmode': 'relative',
  'title': 'Religion Restriction and Harassment'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="5"></a> <br>
# # Histogram
# <font color='red'>
# Lets look at histogram of Countries-Criminal justice in 2011 and 2012. 
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
x2011 = humfree.pf_rol_criminal[humfree.year == 2011]
x2012 = humfree.pf_rol_criminal[humfree.year == 2012]

trace1 = go.Histogram(
    x=x2011,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(191, 200, 06, 0.6)'))
trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name = "2012",
    marker=dict(color='rgba(62, 50, 146, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' Countries-Criminal justice in 2011 and 2012',
                   xaxis=dict(title='Criminal justice'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="6"></a> <br>
# # Scatter Matrix Plots
# <font color='red'>
# Scatter Matrix = it helps us to see covariance and relation between more than 2 features
# <font color='black'>
# * import figure factory as ff
# * create_scatterplotmatrix = creates scatter plot
#     * data2015 = prepared data. It includes Criminal justice, Homicide and Security and safety with index 
#     * colormap = color map of scatter plot
#     * colormap_type = color type of scatter plot
#     * height and weight

# In[ ]:


#preparing Data
df2015 = humfree[humfree.year == 2015]

# import figure factory
import plotly.figure_factory as ff
df = df2015.loc[:,["pf_rol_criminal","pf_ss_homicide", "pf_ss"]]
df["index"] = np.arange(1,len(df)+1)
fig = ff.create_scatterplotmatrix(df, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# <a id="7"></a> <br>
# # Word Cloud
# Not a pyplot but learning it is good for visualization. Lets look at together  which region is most  mentioned .
# * WordCloud = word cloud library that I import at the beginning of kernel
#     * background_color = color of back ground
#     * generate = generates the region name in a word cloud
#     * I have to say that it is only for visualization. Otherwise we can easily find this one any other word's data

# In[ ]:


#preparing Data
x2015 = humfree.region[humfree.year == 2015]
x2015.value_counts()


# In[ ]:


# data prepararion
x2011 = humfree.region[humfree.year == 2011]
plt.subplots(figsize=(10,10))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# <a id="8"></a> <br>
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
x2016 = humfree[humfree.year == 2016]

trace0 = go.Box(
    y=x2016.hf_score,
    name = 'Human Freedom score of countries in 2016',
    marker = dict(
        color = 'rgb(34, 245, 140)',
    )
)
trace1 = go.Box(
    y=x2016.ef_score,
    name = 'Economic Freedom score of countries in 2016',
    marker = dict(
        color = 'rgb(123, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)


# <a id="9"></a> <br>
# # Inset Plots
# <font color='red'>
# Inset Matrix = 2 plots are in one frame
# <font color='black'>

# In[ ]:


x2013 = humfree[humfree.year == 2013]
new_index = (x2013['hf_rank'].sort_values(ascending=True)).index.values
x2013 = x2013.reindex(new_index)
# first line plot
trace1 = go.Scatter(
    x=x2013.hf_rank,  #Human Freedom rank
    y=x2013.hf_score, # Human Freedom Score
    name = "Human Freedom score",
    marker = dict(color = 'rgba(164, 12, 200, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=x2013.hf_rank,    #Human Freedom rank
    y=x2013.ef_score,#Economic Freedom Score
    xaxis='x2',
    yaxis='y2',
    name = "Economic Freedom score",
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
    title = 'Human Freedom vs Economic Freedom scores'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="10"></a> <br>
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
#     
#  # In this data we don't have any column for size so I will use one of them it is only for visualisation.

# In[ ]:


#preparing Data
x2014 = humfree[humfree.year == 2014]
new_index = (x2014['hf_rank'].sort_values(ascending=True)).index.values
x2014 = x2014.reindex(new_index)

# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=x2014.hf_rank,
    y=x2014.ef_regulation_business_bribes,
    z=x2014.ef_regulation_business_licensing,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(145,35,200)',                # set color to an array/list of desired values      
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
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="11"></a> <br>
# # Multiple Subplots
# <font color='red'>
# Multiple Subplots: While comparing more than one features, multiple subplots can be useful.
# <font color='black'>

# In[ ]:


#preparing Data
x2011 = humfree[humfree.year == 2011]
new_index = (x2011['hf_rank'].sort_values(ascending=True)).index.values
x2011 = x2011.reindex(new_index)

trace1 = go.Scatter(
    x=x2011.hf_rank,
    y=x2011.ef_government_consumption,
    name = "Government consumption"
)
trace2 = go.Scatter(
    x=x2011.hf_rank,
    y=x2011.ef_money_inflation,
    xaxis='x2',
    yaxis='y2',
    name = "Inflation"
)
trace3 = go.Scatter(
    x=x2011.hf_rank,
    y=x2011.pf_expression_influence,
    xaxis='x3',
    yaxis='y3',
    name = "Laws and regulations that influence media content"
)
trace4 = go.Scatter(
    x=x2011.hf_rank,
    y=x2011.pf_expression,
    xaxis='x4',
    yaxis='y4',
    name = "Freedom of expression"
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
    title = 'Government consumption,Inflation,Laws and regulations,Freedom of expression VS World Rank of countries'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# # As you realised plotly libraries is very easy(only traces we have to create) and useful (it is talking to us )
# # please write me my errors so I will  learn new things :)
