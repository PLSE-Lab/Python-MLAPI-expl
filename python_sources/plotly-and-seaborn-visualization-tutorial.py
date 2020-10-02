#!/usr/bin/env python
# coding: utf-8

# * [1. Data Exploration](#DataExploration)
# * [2. Data Cleaning](#DataCleaning)
# * [3. Visualisation](#Visualisation)
#     * [3.1 Seaborn](#seaborn)
#         * [3.1.1 Heatmap (Correlation)](#heatmap)
#         * [3.1.2 Scatterplot](#seascatter)
#     * [3.2 Plotly](#plotly)
#         * [3.2.1 Boxplot (Genres VS Revenue)](#boxplot)
#         * [3.2.2 3D Scatterplot (Genre, Revenue VS Popularity)](#scatter)

# <a id='DataExploration'></a>
# 
# ----------
# 
# ## 1.Data Exploration

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


movies =pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


movies.head()


# In[ ]:


movies.info()


# In[ ]:


movies.shape


# In[ ]:


type(movies)


# > Clearly a homepage and tagline have a lot of elements missing

# Looking for how many movies have 0 budget or revenue. Since our analysis is focused on budget and revenue, 0 budget will skew our data

# In[ ]:


# checking for unrealistic values like very low revenue or budget
# movies.groupby('revenue').count()


# In[ ]:


sns.set_style('darkgrid')
sns.distplot(movies['revenue'])
#Kde histogram to see the impact of 0 values on the overall data


# Revenue column has some unrealisticly low values. 1427 movies have made 0 dollors. Other movies have made 1 to a few hundred dollars, which seems unlikely and will not help our analysis.
# 
# Similarly we are going to check for unrealistic values in budget.

# In[ ]:


#movies.groupby('budget').count()


# Lets start by removing all the 0 budget movies. Hopefully this will also remove most of the 0 budget movies

# In[ ]:


#Droppong 0 revenue movies

#df = df.drop(df[df.score<50].index), messed up turned dataframe into None object
#df = df[(df[['A','C']] != 0).all(axis=1)], error list doesnt have attribute all
#df[df.name != 'Tina']
movies = movies[movies['revenue']!=0]
sns.set_style('darkgrid')
sns.distplot(movies['revenue'])


# We can try looking at the year of movies, to see if old movies are messing with out data and focus our analysis within a particular time frame. But first we need to change our release_date into datetime format so we can extrat the year of release easily.

# In[ ]:


movies.release_date = pd.to_datetime(movies.release_date)
movies['year'] = movies.release_date.dt.year
# use a visualisation to see what portion of movies are older that 20 years
sns.set_style('darkgrid')
sns.distplot(movies['year'])


# As these movies data back to 1960's, unless we can account for inflation we will not get a very accurate analysis.

# In[ ]:


movies1 = movies.groupby(['year']).count()


# Two thirds of our movies are concentrared in the 2000's and 2010's. we can concentrate on these years to get a more accurate picture.

# In[ ]:


movies2 = movies[movies['year']>= 2000]
sns.set_style('darkgrid')
sns.distplot(movies2['year'])


# In[ ]:


sns.relplot(x="budget", y="revenue", data= movies2, size ="popularity", hue = 'year', palette = 'Set3',alpha=.7,sizes=(40,1000),height=10)


# In[ ]:


# This is a more uniform dataset
movies2.shape


# <a id='DataCleaning'></a>
# 
# ----------
# 
# ## 2.Data Cleaning

# In[ ]:


#Lets look for outliers
#movies2.groupby('budget').count()


# In[ ]:


# checking the budget to see distribution
sns.set_style('darkgrid')
sns.distplot(movies2['budget'])


# I dont understand how a 0 budget movie can be made. So we decide on a cut off of a 100k dollars as the minimum budget of the movies.

# In[ ]:


movies3 = movies2[movies2['budget']>= 100000]
sns.set_style('darkgrid')
sns.distplot(movies3['budget'])


# In[ ]:


sns.set(style="whitegrid")

ax = sns.boxplot(x=movies3["budget"])


# In[ ]:


#checking the higest revenue movies, creating a sorted dataframe by 
movies_sorted_revenue = movies3.sort_values('revenue', ascending=False)
movies_sorted_revenue.head(30)


# The highest budget movie according to our data set is The Warriors Way with a budget of 425 million dollars. This movie only made 11 million dollars. This data is not accurate and in actuallity The warriors way only costed 42 million to make. So we need to correct this data also.

# In[ ]:


movies4 = movies3[movies3['revenue']>= 100000]
sns.set_style('darkgrid')
sns.distplot(movies4['revenue'])


# In[ ]:


movies4.shape


# In[ ]:


movies3.shape


# In[ ]:


# get the number of missing data points per column
missing_values_count = movies4.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count


# We dont plan to use homepage or tagline.

# ### Stringfield to String (Genre, Keyword, spoken_language)

# In[ ]:


# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = [ 'keywords', 'genres','production_companies','production_countries','spoken_languages']
for feature in features:
    movies4[feature] = movies4[feature].apply(literal_eval)
    
# Returns the top element.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        names =' '.join(names[:1])
    return names

for feature in features:
    movies4[feature] = movies4[feature].apply(get_list)
    
movies4[['title','keywords', 'genres','production_companies','production_countries','spoken_languages']].sample(5)


# In[ ]:


#Checking for correlation in the data
movies.corr()


# <a id='Seaborn'></a>
# 
# ## 3.1 Seaborn Heatmap Correlation

# <a id='heatmap'></a>
# 
# ### 3.1.1 Heatmap Correlation

# In[ ]:


f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(movies.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# There seems to be some association between 1. popularity and vote_count, 2. revenue and vote count and 3. budget and revenue. 
# Surpisingly the relationship between budget and popularity is not strong.

# <a id='seascatter'></a>
# 
# ### 3.1.2 Scatter Plot

# In[ ]:


#df[df.c2.gt(df.c2.quantile(0.8))]
# get top 20% of movies by revenue as the cluster
#df.loc[df['column_name'] == some_value]
movies_top = movies4.loc[movies4['revenue']>=500000000]

# As there is too much data the plot is not meaningful so will try and subset the data


# In[ ]:


sns.relplot(x="budget", y="revenue", data= movies_top, size ="popularity", hue = 'genres', palette = 'Set2',alpha=.7,sizes=(40,1000),height=10)


# <a id='PlotlyVisualisation'></a>
# 
# ----------
# 
# ## 3.2 Plotly 

# In[ ]:


#import plotly
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.display import HTML, Image


# In[ ]:


#find unique values of genre column
#df.B.unique()
movies4['genres'].unique()


# <a id='boxplot'></a>
# ### 3.2.1 Boxplot (Genre VS Revenue)

# In[ ]:


#Interactive boxplots by revenue, hue = genre
action_revenue = movies4[movies4['genres']=='Action']['revenue']
adventure_revenue = movies4[movies4['genres']=='Adventure']['revenue']
drama_revenue = movies4[movies4['genres']=='Drama']['revenue']
comedy_revenue = movies4[movies4['genres']=='Comedy']['revenue']
animation_revenue = movies4[movies4['genres']=='Animation']['revenue']
sciencef_revenue = movies4[movies4['genres']=='Science Fiction']['revenue']

#trace_avg_ph = go.Box(    y=avg_ph,    name = 'Power Hitters',    text = avg_ph.index  )
trace_act_rev = go.Box(y= action_revenue, name ='Action', text=action_revenue.index )
trace_adv_rev = go.Box(y= adventure_revenue, name ='Adventure', text=adventure_revenue.index )
trace_dra_rev = go.Box(y= drama_revenue, name ='Drama', text=drama_revenue.index )
trace_com_rev = go.Box(y= comedy_revenue, name ='Comedy', text=comedy_revenue.index )
trace_ani_rev = go.Box(y= animation_revenue, name ='Animation', text=animation_revenue.index )
trace_sci_rev = go.Box(y= sciencef_revenue, name ='Science Fiction', text=sciencef_revenue.index )

#data = [trace0, trace1, trace2,trace3, trace4, trace5]
#iplot(data)

plo_data = [trace_act_rev, trace_adv_rev, trace_dra_rev, trace_com_rev, trace_ani_rev, trace_sci_rev]
iplot(plo_data)


# Scroll over the graph to see detail. Zoom out using double click. How cool is that :)

# <a id='scatter'></a>
# ### 3.2.2 3D Scatterplot (Genre, Revenue VS Popularity)

# Since there are too many movies which will over crowd the plot, we only want to focus on movies that made more the 500 million dollars. so 

# In[ ]:


# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=movies_top['budget'],
    y=movies_top['revenue'],
    z=movies_top['popularity'],
    mode='markers',
    marker=dict(
        size=12,
        color=movies['vote_count'].values,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.5           # set color to an array/list of desired values      
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

