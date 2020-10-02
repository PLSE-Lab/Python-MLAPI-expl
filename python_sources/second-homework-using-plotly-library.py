#!/usr/bin/env python
# coding: utf-8

# <div id="0">
# # Graphs with Plotly Library 
# 
# In this homework , I will analyze and visualize human behaviours for Black Friday days by using Plotly Library of Python. 
# I will search on products <br><br>
#     
# **Content :** <br>
#     [1. Loading Data and Explanation of Features](#1)<br>
#     [2. Line Charts](#2)<br>
#     [3. Scatter Charts](#3)<br>
#     [4. Bar Charts](#4)<br>
#     [5. Pie Charts](#5)<br>
#     [6. Bubble Charts](#6)<br>
#     [7. Histogram](#7)<br>
#     [8. Word Cloud](#8)<br>
#     [9. Box Plot](#9)<br>
#     [10. Scatter Plot Matrix](#10)<br>
#     [11. Inset Plots](#11)<br>
#     [12. 3D Scatter Plot with Colorscaling](#12)<br>
#     [13. Multiple Subplots](#13)<br>
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# plotly

from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # <div id="1"> 1. Loading Data and Explanation of Features<div/>

# In[ ]:


df = pd.read_csv('../input/BlackFriday.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


#update NaN values with 0 for Product_Category_1 , Product_Category_2 , Product_Category_3

df["Product_Category_1"].fillna(0, inplace=True)
df["Product_Category_2"].fillna(0, inplace=True)
df["Product_Category_3"].fillna(0, inplace=True)


# [**Go To Top**](#0)

# # <div id="2"> 2. Line Charts<div/>
#  
#  Line Charts Example: Male and Female vs buying Product_Category_1
# 
# * Import graph_objs as go
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
# for each gender , sum  Purchase 
df_Gender = df.groupby(['Gender','Product_Category_1'])['Purchase'].sum().reset_index('Gender').reset_index('Product_Category_1')

# add rank column for counts 
df_Gender['Rank'] = df_Gender.groupby('Gender')['Purchase'].rank(ascending=False).astype(int)

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'] 
df_Male = df_Gender[df_Gender['Gender'] =='M'] 

# get only 3 columns 
df_Female = df_Female[['Product_Category_1', 'Purchase','Rank']]
df_Male = df_Male[['Product_Category_1', 'Purchase','Rank']]

#Add category names
df_Female['Product_Category_Name'] = df_Female.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)
df_Male['Product_Category_Name'] = df_Male.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)

df_Female.sort_values(by=['Rank'], inplace = True)
df_Male.sort_values(by=['Rank'], inplace = True)

print(df_Female )


# In[ ]:


# Draw graph 
# Creating trace1
trace1 = go.Scatter(
                    x = df_Female.Rank,
                    y = df_Female.Purchase,
                    mode = "lines",
                    name = "Female Purchase",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df_Female.Product_Category_Name)
# Creating trace2
trace2 = go.Scatter(
                    x = df_Male.Rank,
                    y = df_Male.Purchase,
                    mode = "lines+markers",
                    name = "Male Purchase",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df_Male.Product_Category_Name)
data = [trace1,trace2]
layout = dict(title = 'Male and Female vs buying Product_Category_1',
              xaxis= dict(title= 'Product Purchase Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# [**Go To Top**](#0)

# # <div id="3"> 3. Scatter Charts<div/>
# 
# <font color='red'>
#  Scatter Charts Example: Male and Female vs buying Product_Category_1
# <font color='black'>
# 
# * Import graph_objs as go
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



# creating trace1
trace1 =go.Scatter(
                    x = df_Female.Rank,
                    y = df_Female.Purchase,
                    mode = "markers",
                    name = "Female Purchase",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df_Female.Product_Category_Name)
# creating trace2
trace2 =go.Scatter(
                    x = df_Male.Rank,
                    y = df_Male.Purchase,
                    mode = "markers",
                    name = "Male Purchase",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df_Male.Product_Category_Name)

data = [trace1, trace2]
layout = dict(title = 'Male and Female vs buying Product_Category_1',
              xaxis= dict(title= 'Product Purchase Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Puchase',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)  


# <font color='red'>Second Bar Charts Example: Male and Female vs buying Product_Category_1<br>
# <font color='black'>
# Actually, if you change only the barmode from group to relative in previous example, you achieve what we did here. However, for diversity I use different syntaxes.

# In[ ]:


#second bar chart 

trace1 = {
  'x': df_Female.Product_Category_1,
  'y': df_Female.Purchase,
  'name': 'Female Purchase',
  'type': 'bar'
};
trace2 = {
  'x': df_Male.Product_Category_1,
  'y': df_Male.Purchase,
  'name': 'Male Purchase',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Product Category 1'},
  'barmode': 'relative',
  'title': 'Male and Female vs buying Product_Category_1'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# [**Go To Top**](#0)

# # <div id="4"> 4. Bar Charts<div/>
#     
# <font color='red'>
# First Bar Charts Example:  Male and Female vs buying Product_Category_1
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


# creating trace1
trace1 =go.Bar(
                    x = df_Female.Rank,
                    y = df_Female.Purchase,
                    name = "Female Purchase",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)', line=dict(color='rgb(0,0,0)',width=1.1)),
                    text= df_Female.Product_Category_Name)
# creating trace2
trace2 =go.Bar(
                    x = df_Male.Rank,
                    y = df_Male.Purchase,
                    name = "Male Purchase",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)', line=dict(color='rgb(0,0,0)',width=1.1)),
                    text= df_Male.Product_Category_Name)

data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


#%% second bar chart 
trace1 = {
  'x': df_Female.Product_Category_1,
  'y': df_Female.Purchase,
  'name': 'Female Purchase',
  'type': 'bar'
};
trace2 = {
  'x': df_Male.Product_Category_1,
  'y': df_Male.Purchase,
  'name': 'Male Purchase',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Product Category 1 '},
  'barmode': 'relative',
  'title': 'Male and Female vs buying Product_Category_1'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# [**Go To Top**](#0)

# # <div id="5"> 5. Pie Charts<div/>
#     
# <font color='red'>
# Pie Charts Example: Top 5 Product category1 purchase for men
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



# figure
fig = {
  "data": [
    {
      "values": df_Male.Purchase[df_Male.Rank <=5],
      "labels": df_Male.Product_Category_Name[df_Male.Rank <=5],
      "domain": {"x": [0, .5]},
      "name": "Total Purchase",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],

  "layout": {
        "title":"Top 5 Product category1 purchase for men",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Total Purchase",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)


# [**Go To Top**](#0)

# # <div id="6"> 6. Bubble Charts<div/>
#     
# <font color='red'>
# Bubble Charts Example: Which user bought which  category in top 100 and how many bought
# <font color='black'>
# * x = x axis
# * y = y axis
# * mode = markers(scatter)
# *  marker = marker properties
#     * color = third dimension of plot. Internaltional score
#     * size = fourth dimension of plot. Number of students
# * text: university names    

# In[ ]:



#prepare data 
# filter users who does not buy, means 0 
df_Cat1 = df[df.Product_Category_1 != 0].groupby(['User_ID','Product_Category_1']).count().reset_index('User_ID').reset_index('Product_Category_1')
df_Cat2 = df[df.Product_Category_2 != 0].groupby(['User_ID','Product_Category_1']).count().reset_index('User_ID').reset_index('Product_Category_1')
# Rename column Product_ID as Count
df_Cat1.rename(columns={'Product_ID': 'Count_Prd_Cat1'}, inplace = True)
df_Cat2.rename(columns={'Product_ID': 'Count_Prd_Cat2'}, inplace = True)

# add rank column for counts 
df_Cat1['Rank_Prd_Cat1'] = df_Cat1.groupby('Product_Category_1')['Count_Prd_Cat1'].rank(ascending=False).astype(int)

#filter columns 
df_Cat1 = df_Cat1[['User_ID','Product_Category_1','Count_Prd_Cat1', 'Rank_Prd_Cat1' ]]
df_Cat2 = df_Cat2[['User_ID','Product_Category_1','Count_Prd_Cat2']]
# merge 
df_Users = pd.merge(df_Cat1, df_Cat2, on=['User_ID','Product_Category_1'])

# Add  User_Name  column 
df_Users['User_Name'] = df_Users.apply(lambda row: "Usr_" + row['User_ID'].astype(str), axis=1)
df_Users.sort_values(by=['Product_Category_1'], inplace = True)

df_Users = df_Users[['User_ID', 'User_Name', 'Product_Category_1','Count_Prd_Cat1','Count_Prd_Cat2', 'Rank_Prd_Cat1']][df_Users.Rank_Prd_Cat1 ==1]

df_Users


# In[ ]:


#%% Show first person for each ProductCategory1 and how many they bought .
# define sizes of bubbles for ProductCategory2 to show did they also buy ProductCategory2 
# draw graph 
count_size  = [ float( each) for each in df_Users.Count_Prd_Cat2]
international_color = [float( each) for each in df_Users.User_ID]
    
fig = {
  "data": [
    {
        'y': df_Users.Count_Prd_Cat1,
        'x': df_Users.Product_Category_1,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': count_size,
            'showscale': True
        },
        "text" :  df_Users.User_Name
    },],

  "layout": {
        "title":"Top User For Each Product Category1 and Size for Product Category2",
        "annotations": [
            { "font": { "size": 14},
              "showarrow": False,
              "text": "Product Category 1 ",
                "x": 8,
                "y": -10 
            },
        ]
    }
}
iplot(fig)


# [**Go To Top**](#0)

# # <div id="7"> 7. Histogram<div/>
#     
#  <font color='red'>
# Lets look at histogram of counts purchaced for Product_Category_1 by Female and Male 
# <font color='black'>
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
df_Female = df[df['Gender'] =='F']['Product_Category_1']
df_Male = df[df['Gender'] =='M']['Product_Category_1'] 


# In[ ]:


# Counts for Product_Category_1 for Female and Male
trace1 = go.Histogram(
    x=df_Male,
    opacity=0.75,
    name = "Male",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=df_Female,
    opacity=0.75,
    name = "Female",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' Product_Category_1 counts purchased by Female and Male',
                   xaxis=dict(title='Product Category 1'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# [**Go To Top**](#0)

# # <div id="8"> 8. Word Cloud<div/>
#     
#     Not a pyplot but learning it is good for visualization. Lets look at which ProductId is purchased most
# * WordCloud = word cloud library that I import at the beginning of kernel
#     * background_color = color of back ground
#     * generate = generates the Product_ID list a word cloud

# In[ ]:


#%% World cloud 

Product_ID = df.Product_ID
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(Product_ID))
plt.imshow(wordcloud)
plt.axis('off')
#plt.savefig('graph.png')

plt.show()


# [**Go To Top**](#0)

# # <div id="9"> 9. Box Plot<div/>
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
df_Male = df[df.Gender == 'M']
df_Female = df[df.Gender == 'F']

trace0 = go.Box(
    y=df_Male.Purchase,
    name = 'Total Purchase of Males in Black Friday',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df_Female.Purchase,
    name = 'Total Purchase of Female in Black Friday',
    marker = dict(
        color = 'rgb(12, 128, 12)',
    )
)
data = [trace0, trace1]
iplot(data)


# [**Go To Top**](#0)

# # <div id="10"> 10. Scatter Plot Matrix<div/>
#  <font color='red'>
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
df_Male = df[df.Gender == 'M']

df_Male = df_Male.loc[:100,["Product_Category_1","Product_Category_2", "Product_Category_3"]]
df_Male["index"] = np.arange(1,len(df_Male)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(df_Male, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# [**Go To Top**](#0)

# # <div id="11"> 11. Inset Plots<div/>
#  <font color='red'>
# Inset Matrix = 2 plots are in one frame
# <font color='black'>

# In[ ]:


# prepare data frame 
# for each gender , count  number of Product_Category_1 types 
df_Gender = df.groupby(['Gender','Product_Category_1']).count().reset_index('Gender').reset_index('Product_Category_1')

# add rank column for counts 
df_Gender['Rank'] = df_Gender.groupby('Gender')['Product_ID'].rank(ascending=False).astype(int)

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'] 
df_Male = df_Gender[df_Gender['Gender'] =='M'] 

# get only 3 columns 
df_Female = df_Female[['Product_Category_1', 'Product_ID','Rank']]
df_Male = df_Male[['Product_Category_1', 'Product_ID','Rank']]

# Rename column Product_ID as Count
df_Female.rename(columns={'Product_ID': 'Count'}, inplace = True)
df_Male.rename(columns={'Product_ID': 'Count'}, inplace = True)

df_Female['Product_Category_1'] = df_Female.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)
df_Female.sort_values(by=['Rank'], inplace = True)
df_Male.sort_values(by=['Rank'], inplace = True)
print(df_Female )


# In[ ]:


# Draw graph 
# Creating trace1
trace1 = go.Scatter(
                    x = df_Female.Rank,
                    y = df_Female.Count,
                    mode = "markers",
                    name = "Female Buy",
                    marker = dict(color = 'rgba(256, 0, 0, 0.8)'),
                    text= df_Female.Product_Category_1)
# Creating trace2
trace2 = go.Scatter(
                    x = df_Male.Rank,
                    y = df_Male.Count,
                    xaxis='x2',
                    yaxis='y2',
                    mode = "lines+markers",
                    name = "Male Buy",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df_Male.Product_Category_1)
data = [trace1,trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Male and Female Purchase Count for Product Category 1'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# [**Go To Top**](#0)

# # <div id="12"> 12. 3D Scatter Plot with Colorscaling<div/>
#  <font color='red'>
# 3D Scatter: Sometimes 2D is not enough to understand data. Therefore adding one more dimension increase the intelligibility of the data. Even we will add color that is actually 4th dimension.
# <font color='black'>
# * go.Scatter3d: create 3d scatter plot
# * x,y,z: axis of plots
# * mode: market that is scatter
# * size: marker size
# * color: axis of colorscale
# * colorscale:  actually it is 4th dimension

# In[ ]:


#prepare data frame 
# for each gender , count  number of Product_Category_1 types 
df_Gender = df.groupby(['Gender','Product_Category_1']).count().reset_index('Gender').reset_index('Product_Category_1')

# add rank column for counts 
df_Gender['Rank'] = df_Gender.groupby('Gender')['Product_ID'].rank(ascending=False).astype(int)

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'] 
df_Male = df_Gender[df_Gender['Gender'] =='M'] 

# get only 3 columns 
df_Female = df_Female[['Product_Category_1', 'Product_ID','Rank']]
df_Male = df_Male[['Product_Category_1', 'Product_ID','Rank']]

# Rename column Product_ID as Count
df_Female.rename(columns={'Product_ID': 'Count'}, inplace = True)
df_Male.rename(columns={'Product_ID': 'Count'}, inplace = True)

df_Female['Product_Category_1'] = df_Female.apply(lambda row: "PrdType_" + row['Product_Category_1'].astype(str), axis=1)
df_Female.sort_values(by=['Rank'], inplace = True)
df_Male.sort_values(by=['Rank'], inplace = True)


# In[ ]:


# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=df_Female.Count,
    y=df_Male.Count,
    z=df_Female.Rank,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
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


# [**Go To Top**](#0)

# # <div id="13"> 13 .Multiple Subplots<div/>
#  <font color='red'>
# Multiple Subplots: While comparing more than one features, multiple subplots can be useful.
# <font color='black'>

# In[ ]:


#%%
trace1 = go.Scatter(
    x=df_Female.Rank,
    y=df_Female.Count,
    name = "Female count"
)
trace2 = go.Scatter(
    x=df_Male.Rank,
    y=df_Male.Count,
    xaxis='x2',
    yaxis='y2',
    name = "Male count"
)

data = [trace1, trace2]
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
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    title = 'Product_Category_1 counts purchased by Female and Male'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# [**Go To Top**](#0)

# 
