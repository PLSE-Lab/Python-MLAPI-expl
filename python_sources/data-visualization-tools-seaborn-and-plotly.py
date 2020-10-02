#!/usr/bin/env python
# coding: utf-8

# **CONTENTS**
# 
# **1. SEABORN**
# 
# * [Bar Plot](#1)
# * [Point Plot](#2)
# * [Joint Plot](#3)
# * [Pie Chart](#4)
# * [Lm Plot](#5)
# * [Kde Plot](#6)
# * [Violin Plot](#7)
# * [Heatmap](#8)
# * [Box Plot](#9)
# * [Swarm Plot](#10)
# * [Pair Plot](#11)
# * [Count Plot](#12)
# 
# **2. PLOTLY**
# 
# 1. [Line Charts](#13)
# 1. [Scatter Charts](#14)
# 1. [Bar Charts](#15)
# 1. [Pie Charts](#16)
# 1. [Bubble Charts](#17)
# 1. [Histogram](#18)
# 1. [Box Plot](#19)
# 1. [Scatter Plot Matrix](#20)
# 1. [Inset Plots](#21)
# 1. [Multiple Subplots](#22)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# seaborn library
import seaborn as sns

# plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# matplotlib library
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.info()


# In[ ]:


data.head(10)


# # 1. SEABORN

# <a id="1"></a>
# **1.1) BAR PLOT**

# In[ ]:


data.columns


# In[ ]:


data_70 = data.iloc[:70,:] # first 20 rows of our data


# In[ ]:


# visualization

plt.figure(figsize= (15,10))
sns.barplot(x = data_70['Happiness.Rank'], y = data_70['Freedom'], palette = sns.cubehelix_palette(70))
plt.xticks(rotation = 90) # slope of the words in the x axis 
plt.xlabel('Happiness Rank')
plt.ylabel('Freedom')
plt.title('Happiness Rank vs Freedom')
plt.show()


# <a id="2"></a>
# **1.2)  POINT PLOT # 1**

# In[ ]:


data.columns


# In[ ]:


# visualization
f, ax1 = plt.subplots(figsize=(15,10))
sns.pointplot(x='Happiness.Rank', y='Economy..GDP.per.Capita.', data=data_70, color='black', alpha=0.8)
sns.pointplot(x='Happiness.Rank', y='Freedom', data=data_70, color='red', alpha=1)
plt.text(50,1.65, 'Economy..GDP.per.Capita.', color='black', fontsize=16, style='italic')
plt.text(50,1.77, 'Freedom', color='red',fontsize=16, style='italic')
plt.xlabel('Happines Rank', fontsize=15, color='blue')
plt.ylabel('Values', fontsize=15, color='blue')
plt.title('Economy..GDP.per.Capita. vs Freedom', fontsize=20, color='blue')
plt.grid()


# **1.2) POINT PLOT # 2**

# In[ ]:


# visualization
f, ax1 = plt.subplots(figsize=(15,10))
sns.pointplot(x='Happiness.Rank', y='Economy..GDP.per.Capita.', data=data_70, color='black', alpha=0.8)
sns.pointplot(x='Happiness.Rank', y='Health..Life.Expectancy.', data=data_70, color='red', alpha=1)
plt.text(50,1.65, 'Economy..GDP.per.Capita.', color='black', fontsize=16, style='italic')
plt.text(50,1.74, 'Health..Life.Expectancy.', color='red',fontsize=16, style='italic')
plt.xlabel('Happines Rank', fontsize=15, color='blue')
plt.ylabel('Values', fontsize=15, color='blue')
plt.title('Economy..GDP.per.Capita. vs Health..Life.Expectancy.', fontsize=20, color='blue')
plt.grid()


# <a id="3"></a>
# **1.3) JOINT PLOT (KDE)**

# In[ ]:


data.columns


# In[ ]:


#%%
# joint plot: economy vs health
# kde: kernel density estimation
# pearsonr: if it 1 we have positive correlation, if -1 negative correlation, if zero no correlation  

sns.jointplot(data['Economy..GDP.per.Capita.'], data['Health..Life.Expectancy.'], kind='kde', height=8)
plt.savefig('jointplot.png')
plt.show()


# **1.4) JOINT PLOT (SCATTER)**

# In[ ]:


# we can select different kinds of joint plot
# kind= 'kde', 'scatter', 'reg', 'resit', 'hex' 
# ratio: ratio of the size of the scatter plot to the histogram plot
sns.jointplot(data['Economy..GDP.per.Capita.'], data['Health..Life.Expectancy.'], kind='scatter', size=8, ratio=3, color='r')
plt.show()


# <a id="4"></a>
# **1.5) PIE PLOT**

# In[ ]:


data.columns


# In[ ]:


data['Economy..GDP.per.Capita.'].iloc[0:10]


# In[ ]:


econ = []
for i in range(1,7): 
     econ.append(sum (data['Economy..GDP.per.Capita.'].iloc[((i-1)*10):(i*10)]))
econ


# In[ ]:


# PIE CHART
# Country Economies

# create economy list
econ = []
for i in range(1,7): 
     econ.append(sum (data['Economy..GDP.per.Capita.'].iloc[((i-1)*10):(i*10)]))

# create index
index = np.arange(1,7)

# things to be used in pie plot
labels = index
sizes = econ
colors = ['grey', 'blue', 'red', 'yellow', 'green', 'brown'] # colors of pie chart
explode = [0,0,0,0,0,0]
 
# visualization
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%') # autopct: decimal (after comma put 1 number)
plt.title('Economy Level', color='blue', fontsize=15)
plt.text(1.7,1,'1: Sum of the economy ratios of first 10 countries', fontsize=14)
plt.text(1.7,0.8,'2: Sum of the economy ratios of second 10 countries', fontsize=14)
plt.text(1.7,0.6,'3: Sum of the economy ratios of third 10 countries', fontsize=14)
plt.text(1.7,0.4,'4: Sum of the economy ratios of fourth 10 countries', fontsize=14)
plt.text(1.7,0.2,'5: Sum of the economy ratios of fifth 10 countries', fontsize=14)
plt.text(1.7,0.0,'6: Sum of the economy ratios of sixth 10 countries', fontsize=14)
plt.show()


# <a id="5"></a>
# **1.6) LM PLOT**

# In[ ]:


data.columns


# In[ ]:


# lmplot: shows the result of linear regression within each data set
sns.lmplot(x='Happiness.Score', y='Economy..GDP.per.Capita.', data=data)
plt.show()


# <a id="6"></a>
# **1.7) KDE PLOT**

# In[ ]:


data.columns


# In[ ]:


# kde plot: (kde=kernel density estimation)
plt.subplots(figsize=(15,10))
sns.kdeplot(data['Happiness.Score'], data['Economy..GDP.per.Capita.'], shade=True, cut=3)
plt.xlabel('Happiness', color='purple', fontsize=14)
plt.ylabel('Economy Level', color='purple', fontsize=14)
plt.show()


# <a id="7"></a>
# **1.8) VIOLIN PLOT**

# In[ ]:


data.columns


# In[ ]:


data1 = data.iloc[:,6] # Trust..Government.Corruption.
data2 = data.iloc[:,5] # Economy..GDP.per.Capita.
newdata = pd.concat([data1,data2],axis=1) # create new data
newdata


# In[ ]:


# violin plot
# shows EACH distribution with both violins and points
# use cubehelix to get a custom sequencial palette: google it as seaborn palette 
# it adjusts color of the plots
# violinplot: At the points where the plot is fat, it means we have more points
plt.subplots(figsize=(15,10))
pal = sns.cubehelix_palette(2, rot=-5, dark=3)
sns.violinplot(data=newdata, palette=pal, inner='points') # data uses only numbers but not strings
plt.show()


# <a id="8"></a>
# **1.9) Heatmap**

# In[ ]:


data.columns


# In[ ]:


#correlation map
# Visualization of Economy..GDP.per.Capita., Family and Health..Life.Expectancy. 
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)
plt.show()


# <a id="9"></a>
# **1.10) Box Plot**

# In[ ]:


data.info()


# In[ ]:


data['Happines_level'] = ['HapLev 1' if x<=52 else 'HapLev 2' if x<104 else 'HapLev 3' for x in data['Happiness.Rank']]


# In[ ]:


data['Economy_level'] = ['EconLev 1' if eco>1.31 else 'EconLev 2' if eco>0.99 else 'EconLev 3' 
                         for eco in data['Economy..GDP.per.Capita.']]


# In[ ]:


data.head()


# In[ ]:


# Plot the orbital period with horizontal boxes
sns.boxplot(x="Happines_level", y="Trust..Government.Corruption.", hue='Economy_level', data=data, palette="PRGn")
plt.show()


# <a id="10"></a>
# **1.11) Swarm Plot**

# In[ ]:


sns.swarmplot(x="Happines_level", y="Trust..Government.Corruption.",hue="Economy_level", data=data)
plt.show()


# <a id="11"></a>
# **1.12) Pair Plot**

# In[ ]:


data.columns


# In[ ]:


datam = data.iloc[:, [5,6,7]]


# In[ ]:


datam.head()


# In[ ]:


# pair plot
sns.pairplot(datam)
plt.show()


# <a id="12"></a>
# **1.13) Count Plot**

# In[ ]:


data.Economy_level.value_counts()


# In[ ]:


sns.countplot(data.Economy_level)
# sns.countplot(data.Happines_level)
plt.title("Economy Level",color = 'blue',fontsize=15)


# **2. PLOTLY**

# <a id="13"></a>
# **2.1) Line Chart**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = data['Happiness.Rank'],
                    y = data['Economy..GDP.per.Capita.'],
                    mode = "lines",
                    name = "Economy GDP per Capita",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= data.Country)
# Creating trace2
trace2 = go.Scatter(
                    x = data['Happiness.Rank'],
                    y = data['Health..Life.Expectancy.'],
                    mode = "lines+markers",
                    name = "Health Life Expectancy.",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= data.Country)
data = [trace1, trace2]
layout = dict(title = 'Economy and Health Life Expectancy vs Country Ranks',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
plt.savefig('line.png')
plt.show()


# <a id="14"></a>
# **2.2) Scatter**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.head()


# In[ ]:


# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter(
                    x = data['Happiness.Rank'],
                    y = data['Economy..GDP.per.Capita.'],
                    mode = "markers",
                    name = "Economy",
                    marker = dict(color = 'rgba(0, 128, 255, 0.8)'),
                    text= data['Country'])
# creating trace2
trace2 =go.Scatter(
                    x = data['Happiness.Rank'],
                    y = data['Trust..Government.Corruption.'],
                    mode = "markers",
                    name = "Trust-Goverment",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= data['Country'])

data = [trace1, trace2]
layout = dict(title = 'Economy and Trust-Goverment vs Happiness Rank',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id="15"></a>
# **2.3) Bar Chart #1**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.head()


# In[ ]:


# prepare data frames
data_3 = data.iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = data_3.Country,
                y = data_3['Generosity'],
                name = "Generosity",
                marker = dict(color = 'rgba(200, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_3['Happiness.Rank'])
# create trace2 
trace2 = go.Bar(
                x = data_3.Country,
                y = data['Family'],
                name = "Family",
                marker = dict(color = 'rgba(150, 50, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_3['Happiness.Rank'])
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **2.2 Bar Chart #2**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


# prepare data frames
data_3 = data.iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = data_3.Country,
                y = data_3['Generosity'],
                name = "Generosity",
                marker = dict(color = 'rgba(200, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_3['Happiness.Rank'])
# create trace2 
trace2 = go.Bar(
                x = data_3.Country,
                y = data['Family'],
                name = "Family",
                marker = dict(color = 'rgba(150, 50, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_3['Happiness.Rank'])
data = [trace1, trace2]
layout = go.Layout(barmode = "relative")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="16"></a>
# **2.3) Pie Chart**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.head()


# In[ ]:


data_7 = data.iloc[:7,:]
data_7


# In[ ]:


pie1 = data_7['Economy..GDP.per.Capita.']
labels = data_7['Country']
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Economy rates of some Countries",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Economy rates of first 7 Countries",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Economy Level",
                "x": 0.17,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)


# <a id="17"></a>
# **2.4) Bubble Chart**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.head()


# In[ ]:


# data preparation
data_20 = data.iloc[:20,:]
econ_size  = data_20['Economy..GDP.per.Capita.']*30
family_color = data_20['Family']
data = [
    {
        'y': data_20['Freedom'],
        'x': data_20['Happiness.Rank'],
        'mode': 'markers',
        'marker': {
            'color': family_color,
            'size': econ_size,
            'showscale': True
        },
        "text" :  data_20.Country    
    }
]
layout = dict(title='Freedom vs Happiness Rank: Size=Economy, Color=Family',
              xaxis=dict(title='Happiness Rank'),
              yaxis=dict(title='Freedom'))
fig = go.Figure(data=data,layout=layout)
iplot(fig)


# <a id="18"></a>
# **2.5) Histogram**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


# prepare data
x1 = data['Economy..GDP.per.Capita.']
x2 = data['Family']

trace1 = go.Histogram(
    x=x1,
    opacity=0.75,
    name = "Economy",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=x2,
    opacity=0.75,
    name = "Family",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title='Economy and Family',
                   xaxis=dict(title='Economy and Family Rates'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="19"></a>
# **2.6) Box Plot**

# In[ ]:


data = pd.read_csv('../input/2017.csv')


# In[ ]:


data.columns


# In[ ]:


trace0 = go.Box(
    y=data['Economy..GDP.per.Capita.'],
    name = 'Economy',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=data['Trust..Government.Corruption.'],
    name = 'Trust to Governmetn',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)


# <a id="20"></a>
# **2.7) Scatter Matrix Plot**

# In[ ]:


data = pd.read_csv('../input/2017.csv')
data.columns


# In[ ]:


# import figure factory
import plotly.figure_factory as ff

data_matrix = data.loc[:,["Health..Life.Expectancy.","Freedom", "Happiness.Rank", 'Economy..GDP.per.Capita.']]
data_matrix["index"] = np.arange(1,len(data_matrix)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data_matrix, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# <a id="21"></a>
# **2.8) Inset Plots**

# In[ ]:


data = pd.read_csv('../input/2017.csv')
data.columns


# In[ ]:


# first line plot
x=data['Happiness.Rank']
trace1 = go.Scatter(
    x=x,
    y=data['Economy..GDP.per.Capita.'],
    name = "Economy",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=x,
    y=data['Trust..Government.Corruption.'],
    xaxis='x2',
    yaxis='y2',
    name = "Trust Government",
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
    title = 'Economy and Trust Government vs Happiness Rank'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="22"></a>
# **2.9 Multiple Subplots**

# In[ ]:


data = pd.read_csv('../input/2017.csv')
data.columns


# In[ ]:


x=data['Happiness.Rank']
trace1 = go.Scatter(
    x=x,
    y=data['Economy..GDP.per.Capita.'],
    name = "economy"
)
trace2 = go.Scatter(
    x=x,
    y=data['Family'],
    xaxis='x2',
    yaxis='y2',
    name = "family"
)
trace3 = go.Scatter(
    x=x,
    y=data['Health..Life.Expectancy.'],
    xaxis='x3',
    yaxis='y3',
    name = 'health'
)
trace4 = go.Scatter(
    x=x,
    y=data['Freedom'],
    xaxis='x4',
    yaxis='y4',
    name = "freedom"
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
    title = 'economy-health-family-freedom vs happiness rank'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# # CONCLUSION
# * if you find it helpful, tank you for upvotes
# * for a better one you can read kernels of DATAI TEAM
