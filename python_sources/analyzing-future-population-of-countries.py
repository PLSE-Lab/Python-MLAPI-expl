#!/usr/bin/env python
# coding: utf-8

# 1. Introduction: (#1)
# 2. Loading Data and Explanation of Features: (#2)
# 3. Exploratory Data Analysis (EDA): (#3)
# 4. Applying Population Growth Models: (#4)
# 5. CONCLUSION: (#5)

# <a id="1"></a> 
# **1. Introduction**
# 
# Hello everyone!  In this kernel we will be working on Countries of the World Dataset . we will analyze Population of Countries with correlated features and also we will apply population growth models with visualisation.
# 
# The datasets consist of several  independent variables  include:
# 
# * Country
# * Region
# * Population
# * Area (sq. mi.)
# * Pop. Density (per sq. mi.)
# * Coastline (coast/area ratio)
# * Net migration
# * Infant mortality (per 1000 births)
# * GDP ($ per capita)
# * Literacy (%)
# * Phones (per 1000)
# * Arable (%)
# * Crops (%)
# * Other (%)
# * Climate
# * Birthrate
# * Deathrate
# * Agriculture
# * Industry
# * Service
# 
# We are going to use some of the variables which is about population growth.

# <a id="2"></a> 
# **2. Loading Data and Explanation of Features**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# plotly
import plotly.plotly as py
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
#seaborn
import seaborn as sns
# matplotlib
import matplotlib.pyplot as plt
#missingno
import missingno as msno
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))


# In[ ]:


data=pd.read_csv("../input/countries of the world.csv")
data.head(10)


# In[ ]:


data.info()


# As we can see, our data has NaN values and also some features' types are object that it is necessary to be changed.

# In[ ]:


data.replace(to_replace=r",",value=".",regex=True,inplace=True) # replace "," to "."


# In[ ]:


data.columns[4:] # select features which types are going to change


# In[ ]:


for each in data.columns[4:]:
    data[each]=data[each].astype("float") # change object to float
data.dtypes


# Now we are going to missingno library for see the Nan values as a alternative to data.isna().any() 

# In[ ]:


msno.matrix(data)
plt.show()


# In[ ]:


data.isna().sum() # sum. of the Nan values at each feature


# Also we are going to look for zero values because some features that having zero value is illogical.

# In[ ]:


data[data==0].count() # sum. of the zeros at each feature


# In[ ]:


data.columns[7:] # select the features which is going to change


# In[ ]:


for each in data.columns[7:]:
    data[each].replace(to_replace=0,value="NaN",inplace=True) # replace zeros to NaN values
for each in data.columns[7:]:
    data[each]=data[each].astype("float") # again making objects to float 
data.dropna(inplace=True) #dropping NaN values
data.index = range(len(data.index)) # rearange index numbers
data.info()


# As we cane see above, our data finally looks clear and sensible. Now we can proceed to visualisation data for understand better.  

# <a id="3"></a> 
# **3. Exploratory Data Analysis (EDA)**

# Lets see how many countries in each region by our data.

# In[ ]:


trace = [go.Bar(
            x=data.Region.value_counts().index,
            y=data.Region.value_counts().values,
            text=data.Region.value_counts().values,
            hoverinfo = 'text',
            textposition = 'auto',
            marker = dict(color = 'rgba(253,174,97, 0.5)',
                             line=dict(color='rgb(0,200,200)',width=1.5)),
    )]

layout = dict(
    title = 'Number of Countries by Region',
)
fig = go.Figure(data=trace, layout=layout)
iplot(fig)


# And we are going to count population of each region.

# In[ ]:


data.groupby("Region")["Population"].sum() 


# Also we are going to visualise rate of populations by regions to understand population details.

# In[ ]:


fig = {
  "data": [
    {
      "values": data.groupby("Region")["Population"].sum().values,
      "labels": data.groupby("Region")["Population"].sum().index,
      "domain": {"x": [0, .5]},
      "name": "Population Rate",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Rate of Population by Regions",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Regions",
                "x": 0.20,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)


# In[ ]:


trace = dict(type='choropleth',
            locations = data.Country,
            locationmode = 'country names', z = data.Population,
            text = data.Country, colorbar = {'title':'Population'},
            colorscale = 'Hot', reversescale = True)

layout = dict(title='Population of Countries',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [trace],layout = layout)
iplot(choromap,validate=False)


# If we analyze above these three plot, we can see which region has more country,  which is the most populated and where they located at. Now we are going to proceed to apply growth models on our data.  

# <a id="4"></a> 
# **4. Applying Population Growth Models**

# Firstly we have two growth models which are needed to be explained.

# **Exponential Function Constant Growth Rate Model:**
# 
# f(t) = a*(b+1)^t
# 
# f(t) = population after t years
# 
# a = initial value
# 
# b = base or growth factor
# 
# t = time in years
# 
# This exponential model can be used to predict population during a period when the population growth rate remains constant.

# **Exponential Function Continuous Change Model:**
# 
# A(t) = P*e^r*t
# 
# A(t) = amount of population after t years
# 
# P = initial Population
# 
# e = exponential constant
# 
# r = annual growth rate
# 
# t = time in years
# 
# This exponential model can be used to predict population during a period when the growth of a population is continuous.

# Secondly we are going to apply them on countries which selected later. Before this we need to find growth rate which we will use in models. Lets find our groth rate:

# Gr=(B+N-D)/10
# 
# Gr=Growth rate %
# 
# B=Birthrate
# 
# N=Net migration
# 
# D=Deathrate
# 
# All of the parameters can be selected from our data.

# In[ ]:


def rate(row):
    return (row["Birthrate"]+row["Net migration"]-row["Deathrate"])/10
data["Growth Rate%"]=data.apply(rate,axis=1) # crating new column


# We calculated Growth rate. Now its time to apply our models. But we have too many countries so I selected the most populated by each region for making it simple.

# In[ ]:


data.groupby("Region")["Population"].max()
max_pop=data.loc[data.Population.isin(data.groupby("Region")["Population"].max().values)]
max_pop


# Lets apply our models. We are going to find next year population of countries we selected.

# In[ ]:


def linear(row):
    return row["Population"]*((row["Growth Rate%"]/100)+1)
max_pop["Next Year Pop."]=max_pop.apply(linear,axis=1) 


# In[ ]:


import math # math library for calculation
def expo(row):
    return (row["Population"]*(math.exp(row["Growth Rate%"]/100)))
max_pop["Next Year Pop. Exp."]=max_pop.apply(expo,axis=1)
max_pop[["Country","Population","Next Year Pop.","Next Year Pop. Exp."]]


# Now lets visualise them to understand easly.

# In[ ]:


trace1 = go.Bar(
                x = max_pop.Country,
                y = max_pop.Population,
                name = "Current Population",
                marker = dict(color = 'rgba(0, 255, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
trace2 = go.Bar(
                x = max_pop.Country,
                y = max_pop["Next Year Pop."],
                name = "Next Year Pop.",
                marker = dict(color = 'rgba(255, 255, 0, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
data1 = [trace1, trace2]
layout = go.Layout(barmode = "group",title="Next Year Population by Constant Growth")
fig = go.Figure(data = data1, layout = layout)
iplot(fig)


# In[ ]:


trace1 = go.Bar(
                x = max_pop.Country,
                y = max_pop.Population,
                name = "Current Population",
                marker = dict(color = 'rgba(0, 255, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
trace2 = go.Bar(
                x = max_pop.Country,
                y = max_pop["Next Year Pop. Exp."],
                name = "Next Year Pop.",
                marker = dict(color = 'rgba(255, 0, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
data1 = [trace1, trace2]
layout = go.Layout(barmode = "group",title="Next Year Population by Continuous Growth")
fig = go.Figure(data = data1, layout = layout)
iplot(fig)


# Both calculation show us calculated populations are very close. How about finding populations after 5 years later.

# In[ ]:


def linear5(row):
    return row["Population"]*math.pow(((row["Growth Rate%"]/100)+1),5)
max_pop["After 5 Year Pop."]=max_pop.apply(linear5,axis=1)


# In[ ]:


import math
def expo5(row):
    return row["Population"]*math.pow((math.exp(row["Growth Rate%"]/100)),5)
max_pop["After 5 Year Pop. Exp."]=max_pop.apply(expo5,axis=1)
max_pop[["Country","Population","After 5 Year Pop.","After 5 Year Pop. Exp."]]


# In[ ]:


trace1 = go.Bar(
                x = max_pop.Country,
                y = max_pop["After 5 Year Pop."],
                name = "Constant Growth",
                marker = dict(color = 'rgba(500,100,150, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
trace2 = go.Bar(
                x = max_pop.Country,
                y = max_pop["After 5 Year Pop. Exp."],
                name = "Continuous Growth",
                marker = dict(color = 'rgba(150,100,500, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
data1 = [trace1, trace2]
layout = go.Layout(barmode = "group",title="Populations After 5 Year")
fig = go.Figure(data = data1, layout = layout)
iplot(fig)


# We have found the populations of next year and 5 years later. Even 5 years later our models calculated population very closely to each other Lets see the features that correlated with our growth rate.

# In[ ]:


f, ax = plt.subplots(figsize=(16, 16))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data.corr(), cmap=cmap, vmax=.3, center=0,square=True,annot=True, linewidths=.5, cbar_kws={"shrink": .5},fmt= '.1f')
plt.show()


# I am going to select the features which are correlated moderately or higher. But I left some features even though it is correlated because these features were used for calculation of growth rate. After selection lets look at correlation matrix.

# In[ ]:


data2=data.loc[:,["GDP ($ per capita)","Literacy (%)","Climate","Agriculture","Industry","Service","Growth Rate%"]]
data2["index"]=np.arange(1,len(data2)+1)
fig = ff.create_scatterplotmatrix(data2, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',title="Growth Factors",
                                  height=1200, width=1200)
iplot(fig)


# We can inspect correlation of each feature by each other and how they scattered.
# 
# Lastly we are going to find Physiological Intensity for slected countries. For those who dont know what Physiological Intensity means, it is a density of population on crop area which is a useful feature that how many people feed or benefit  from crop area in his/her country.
# 
# Pi=Pd/Cr*100
# 
# Pi=Physiological Intensity
# 
# Pd=Population Density
# 
# Cr=Rate of Crops

# In[ ]:


def Intensity(row):
    return (row["Pop. Density (per sq. mi.)"]/row["Crops (%)"])*100
max_pop["Physiological Intensity(sq.mi.)"]=max_pop.apply(Intensity,axis=1)
max_pop[["Country","Physiological Intensity(sq.mi.)"]]


# Lets  see Physiological Intensity versus population by population density and crops' rate:

# In[ ]:


trace0 = go.Scatter(
    x=max_pop["Physiological Intensity(sq.mi.)"],
    y=max_pop["Population"],
    text=max_pop.Country ,
    mode='markers',
    marker=dict(
        colorbar = {'title':'Crops (%)'},
        color=max_pop["Crops (%)"],
        size=max_pop["Pop. Density (per sq. mi.)"],
        showscale=True
    )
)

data3 = [trace0]
layout = go.Layout(
    title='Physiological Intensity v. Population by v. Pop. Density by Crops% ',
    xaxis=dict(
        title='Physiological Intensity',
    ),
    yaxis=dict(
        title='Population',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig = go.Figure(data = data3, layout = layout)
iplot(fig)


# We can see Germany has the most Physiological Intensity due to high population and low crop area.

# <a id="5"></a> 
# **5. CONCLUSION**

# Finally we analyzed our data by population and its correlated features, applied our models and also visualisated what we found.Do you think which growth' model is suitable for caluculating future populations. And what can we find different features from population.
# 
# * If you like it, thank you for you upvotes.
# * If you have any question, I will happy to hear it
# 
# https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners for plotly tutorial
