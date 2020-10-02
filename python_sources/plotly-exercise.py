#!/usr/bin/env python
# coding: utf-8

# Content:
# 
# * Loading Data and Explanation of Features
# * Line Charts
# * Scatter Charts
# * Bar Charts
# * Pie Charts
# * Bubble Charts
# * Histogram
# * Word Cloud
# * Box Plot
# * Scatter Plot Matrix
# * Inset Plots
# * 3D Scatter Plot with Colorscaling
# * Multiple Subplots

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#matplotlib
import matplotlib.pyplot as plt
#plotly
import plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# word cloud 
from wordcloud import WordCloud


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Loading Data and Explanation of Features**

# In[ ]:


#read data
timesdata = pd.read_csv("../input/world-university-rankings/timesData.csv")


# In[ ]:


timesdata.head()


# In[ ]:


timesdata.info()


# **Line Chart**

# We will compare teaching and citation as world universities rank as
# 
# -Import graph_objs as go
# 
# -Creating traces
# 
# * x = x axis
# * y = y axis
# * mode = type of plot like marker, line or line + markers
# * name = name of the plots
# * marker = marker is used with dictionary.
#     *  color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
# * text = The hover text (hover is curser)
# 
# -data = is a list that we add traces into it
# 
# -layout = it is dictionary.
# 
# * title = title of layout
# * x axis = it is dictionary
#     * title = label of x axis
#     * ticklen = length of x axis ticks
#     * zeroline = showing zero line or not
#     
# -fig = it includes data and layout
# 
# -iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


#create new data frame
df = timesdata.iloc[:100,:]

# import graph objects as "go"
import plotly.graph_objs as go

#Creating trace
trace1 = go.Scatter( x = df.world_rank,
                     y = df.citations,
                     mode = "lines",
                     name = "Citations",
                     marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                     text= df.university_name
                    )
#Creating trace
trace2 = go.Scatter( x = df.world_rank,
                     y = df.teaching,
                     mode = "lines+markers",
                     name = "Teaching",
                     marker = dict(color = 'rgba(106, 112, 200, 0.8)'),
                     text= df.university_name
                    )
data = [trace1,trace2]
layout = dict (title = "Citation and Teaching vs World Rank of Top 100 Universities", xaxis = dict (title ="World Rank",ticklen= 5,zeroline= False),
             yaxis = dict(title = "Rate", ticklen = 5,zeroline = False))
fig = dict (data = data , layout = layout)
iplot(fig)


# **Scatter Plot**

# We will compare citation and world universities rank as 2014,2015,2016 years as
# 
# -Import graph_objs as go
# 
# -Creating traces
# 
# * x = x axis
# * y = y axis
# * mode = type of plot like marker, line or line + markers
# * name = name of the plots
# * marker = marker is used with dictionary.
#     * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
# * text = The hover text (hover is curser)
# 
# -data = is a list that we add traces into it
# 
# -layout = it is dictionary.
# 
# * title = title of layout
# * x axis = it is dictionary
#     * title = label of x axis
#     * ticklen = length of x axis ticks
#     * zeroline = showing zero line or not
# * y axis = it is dictionary and same with x axis
# 
# -fig = it includes data and layout
# 
# -iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frames
df2014 = timesdata[timesdata.year == 2014].iloc[:100,:]
df2015 = timesdata[timesdata.year == 2015].iloc[:100,:]
df2016 = timesdata[timesdata.year == 2016].iloc[:100,:]
# import graph objects as "go"
import plotly.graph_objs as go
#Creating trace
trace1 = go.Scatter(x =df2014.world_rank, 
                    y = df2014.citations,
                   mode = "markers",
                   name = "2014",
                   marker = dict (color = "rgba(255,128,255,0.8)"),
                   text = df2014.university_name)
#Creating trace
trace2 = go.Scatter(x =df2015.world_rank, 
                    y = df2015.citations,
                   mode = "markers",
                   name = "2015",
                   marker = dict (color = "rgba(28,239,9,0.8)"),
                   text = df2015.university_name)
#Creating trace
trace3 = go.Scatter(x =df2016.world_rank, 
                    y = df2016.citations,
                   mode = "markers",
                   name = "2015",
                   marker = dict (color = "rgba(150,30,40,0.8)"),
                   text = df2015.university_name)
data2 = [trace1 ,trace2, trace3]
layout2 = dict ( title = "Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years",
              xaxis = dict ( title = "World Rank", ticklen = 5 , zeroline = False),
              yaxis = dict (title = "Citation", ticklen = 5 , zeroline = False))
fig = (dict(data = data2, layout = layout2))
iplot(fig)


# **Bar Chart**

# We will compare citations and teaching of top 3 universities in 2014
# 
# -Import graph_objs as go
# 
# -Creating traces
# 
# * x = x axis
# * y = y axis
# * mode = type of plot like marker, line or line + markers
# * name = name of the plots
# * marker = marker is used with dictionary.
#     * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#     * line = It is dictionary. line between bars
#         * color = line color around bars
# * text = The hover text (hover is curser)
# 
# -data = is a list that we add traces into it
# 
# -layout = it is dictionary.
#    * barmode = bar mode of bars like grouped
#    
# -fig = it includes data and layout
# 
# -iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


#Style1
# prepare data frames
df2014 = timesdata[timesdata.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
#Creating trace
trace1 = go.Bar (x =df2014.university_name,
                y =df2014.citations,
                marker = dict(color = "rgba(249,13,9,0.8)",
                             line = dict (color = "rgb(0,0,0)",width = 1.5)),
                text = df2014.country)
#Creating trace
trace2 = go.Bar (x =df2014.university_name,
                y =df2014.teaching,
                marker = dict(color = "rgba(226,223,9,0.8)",
                             line = dict (color = "rgb(0,0,0)",width = 1.5)),
                text = df2014.country)
data3 = [trace1,trace2]
# you dont need to write 1,2,3 line
layout3 = go.Layout(title = "citations and teaching of top 3 universities in 2014", #1
                   xaxis = dict( title = "University Name", ticklen = 5 , zeroline = False), #2
                   yaxis = dict( title = "Rate", ticklen = 5 , zeroline = False), #3
                   barmode = "group")
fig =dict(data=data3, layout=layout3)
iplot(fig)


# In[ ]:


#Style2
# prepare data frames
df2014 = timesdata[timesdata.year == 2014].iloc[:3,:]
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
data4 = [trace1, trace2];
layout4 = {
  'xaxis': {'title': 'Top 3 universities'},
  'barmode': 'relative',
  'title': 'citations and teaching of top 3 universities in 2014'
};
fig = go.Figure(data = data4, layout = layout4)
iplot(fig)


# In[ ]:


# Or
# prepare data frames
df2014 = timesdata[timesdata.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
#Creating trace6
trace6 = go.Bar (x =df2014.university_name,
                y =df2014.citations,
                marker = dict(color = "rgba(249,13,9,0.8)",
                             line = dict (color = "rgb(0,0,0)",width = 1.5)),
                text = df2014.country)
#Creating trace7
trace7 = go.Bar (x =df2014.university_name,
                y =df2014.teaching,
                marker = dict(color = "rgba(226,223,9,0.8)",
                             line = dict (color = "rgb(0,0,0)",width = 1.5)),
                text = df2014.country)
data3 = [trace6,trace7]
# you dont need to write 1,2,3 line
layout3 = go.Layout(title = "citations and teaching of top 3 universities in 2014", #1
                   xaxis = dict( title = "University Name", ticklen = 5 , zeroline = False), #2
                   yaxis = dict( title = "Rate", ticklen = 5 , zeroline = False), #3
                   barmode = "relative")
fig =dict(data=data3, layout=layout3)
iplot(fig)


# **Pie Chart**

# We will compare students rate of top 7 universities in 2016
# 
# -fig: create figures
# 
# * data: plot type
#     * values: values of plot
#     * labels: labels of plot
#     * name: name of plots
#     * hoverinfo: information in hover
#     * hole: hole width
#     * type: plot type like pie
# * layout: layout of plot
#     * title: title of layout
#     * annotations: font, showarrow, text, x, y

# In[ ]:


# data preparation
df2016 = timesdata[timesdata.year == 2016].iloc[:7,:]
df2016.info()


# In[ ]:


df2016.head()
#we should do 2,243 > 2.243


# In[ ]:


pie1 = df2016.num_students
pie1_list = [float(each.replace(",",".")) for each in df2016.num_students] # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = df2016.university_name


# In[ ]:


trace =go.Pie(values =pie1_list,
              labels = labels,
              name = "Number Of Students Rates",
              hoverinfo = "label+percent+name",
              hole = 0.3,
              )
layout = go.Layout(title = "Universities Number of Students rates",
                   annotations=[dict(text="Number of Students",
                                     x=0.9,
                                     y=1.1,
                                     font_size=20,
                                     showarrow=False)])
fig = dict(data = trace, layout = layout)
iplot(fig)


# In[ ]:


#or 
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Universities Number of Students rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Students",
                "x": 0.70,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)


# **Buble Chart**

#  We will compare university world rank (first 20) vs teaching score with number of students(size) and international score (color) in 2016
# 
# * x = x axis
# * y = y axis
# * mode = markers(scatter)
# * marker = marker properties
#     * color = third dimension of plot. Internaltional score
#     * size = fourth dimension of plot. Number of students
# * text: university names

# In[ ]:


# data preparation
df2016 = timesdata[timesdata.year == 2016].iloc[:20,:]
num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]
international_color = [float(each) for each in df2016.international]
trace = go.Scatter(x =df2016.world_rank, 
                    y = df2016.teaching,
                   mode = "markers",
                   name = "2016",
                   marker = dict (color =international_color, size = num_students_size , showscale = True ),
                   text = df2016.university_name)
layout = dict ( title = "University world rank vs teaching score with number of students(size) and international score (color) in 2016",
              xaxis = dict ( title = "World Rank", ticklen = 5 , zeroline = False),
              yaxis = dict (title = "Teaching", ticklen = 5 , zeroline = False))
fig = (dict(data = trace, layout = layout))
iplot(fig)


# In[ ]:


#or
df2016 = timesdata[timesdata.year == 2016].iloc[:20,:]
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


# **Histogram**

# * trace1 = first histogram
#     * x = x axis
#     * y = y axis
#     * opacity = opacity of histogram
#     * name = name of legend
#     * marker = color of histogram
# * trace2 = second histogram
# * layout = layout
#     * barmode = mode of histogram like overlay. Also you can change it with stack

# In[ ]:


x2011 = timesdata.student_staff_ratio[timesdata.year == 2011]
x2012 = timesdata.student_staff_ratio[timesdata.year == 2012]
trace10 = go.Histogram(x = x2011,
                      opacity = 0.8,
                      name = "2011",
                      marker = dict (color = "rgb(171, 50, 96)"))
trace11 = go.Histogram(x = x2012,
                      opacity = 0.8,
                      name= "2012",
                      marker = dict ( color = "rgb(12, 50, 196)"))
data6 = [trace10, trace11]
layout6 = go.Layout(barmode = "overlay",
                   title = "students-staff ratio in 2011 and 2012'",
                   xaxis = dict(title = "students-staff ratio"),
                   yaxis = dict(title = "Frequency"))
fig = go.Figure(data = data6, layout = layout6)
iplot(fig)


# **Word Cloud**

# * background_color = color of back ground
# * generate = generates the country name list(x2011) a word cloud

# In[ ]:


x2011forwordcloud = timesdata.country[timesdata.year == 2011]
plt.subplots(figsize =(10,10))
wordcloud = WordCloud (background_color = "White",
                       #max_font_size = 50, we can change font size and max words
                       #max_words = 100,
                      width = 512,
                      height= 384,
                      ).generate(" ".join(x2011forwordcloud))
plt.imshow(wordcloud,
           interpolation="bilinear")
plt.axis('off')
plt.show()


# **Box Plot**

# -Median (50th percentile) = middle value of the data set. Sort and take the data in the middle. It is also called 50% percentile that is 50% of data are less that median(50th quartile)(quartile)
# 
#    * 25th percentile = quartile 1 (Q1) that is lower quartile
#    * 75th percentile = quartile 3 (Q3) that is higher quartile
#    * height of box = IQR = interquartile range = Q3-Q1
#    * Whiskers = 1.5 * IQR from the Q1 and Q3
#    * Outliers = being more than 1.5*IQR away from median commonly.
#    
# -trace = box
# 
#    * y = data we want to visualize with box plot
#    * marker = color

# In[ ]:


x2015 = timesdata[timesdata.year == 2015]

trace1 = go.Box (y = x2015.total_score,
                 name = "total score of universities in 2015",
                 marker = dict(color = "rgb(12, 12, 140)"))

trace2 = go.Box (y = x2015.research,
                 name = "research of universities in 2015",
                 marker = dict(color = "rgb(12, 128, 128)"))
data = [trace1,trace2]
iplot(data)


# **Scatter Matrix Plot**

# it helps us to see covariance and relation between more than 2 features
# 
# * import figure factory as ff
# * create_scatterplotmatrix = creates scatter plot
#     * data2015 = prepared data. It includes research, international and total scores with index from 1 to 401
#     * colormap = color map of scatter plot
#     * colormap_type = color type of scatter plot
#     * height and weight

# In[ ]:


# import figure factory
import plotly.figure_factory as ff

dataframe = timesdata[timesdata.year == 2011]
data2015 = dataframe.loc[:,["research","international", "total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# **3D Scatter Plot with Colorscaling**

# * go.Scatter3d: create 3d scatter plot
# * x,y,z: axis of plots
# * mode: market that is scatter
# * size: marker size
# * color: axis of colorscale
# * colorscale: actually it is 4th dimension

# In[ ]:



# create trace  that is 3d scatter
trace = go.Scatter3d(
    x=dataframe.world_rank,
    y=dataframe.research,
    z=dataframe.citations,
    mode='markers',
    marker=dict(
        size=10,
        color= "rgb(255,0,0)" )                # set color to an array/list of desired values
    )
        


layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0),
                   )

fig = go.Figure(data=trace, layout=layout)
iplot(fig)


# **Multiple Subplots**

# While comparing more than one features, multiple subplots can be useful.

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
data = [trace1, trace2, trace3, trace4]
fig = go.Figure(data=data, layout = layout)
iplot(fig)

