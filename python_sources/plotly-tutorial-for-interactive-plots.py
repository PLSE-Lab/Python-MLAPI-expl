#!/usr/bin/env python
# coding: utf-8

# ### <center>We will use the popular open source library Plotly for data visualization.</center>

# Here you will see how to draw interactive graphics easily and with a minimum of code. Plotly is an extremely powerful tool and it is impossible to cover all its features at once, so I'll show you how to build the most relevant and interesting graphics.

# - <a href='#pie'>Pie</a>  
# - <a href='#bar'>Bar</a>  
# - <a href='#scatter'>Scatter</a> 
# - <a href='#box'>Box</a>  
# - <a href='#choropleth'>Choropleth</a>  

# First, if you don't already have Plotly installed, run:

# In[ ]:


#!pip install plotly


# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.offline as py
import warnings
import pycountry
warnings.filterwarnings('ignore')

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[ ]:


PATH = '../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv'
data = pd.read_csv(PATH)
data.head(3)


# The dataset has the following features:
# 
# - __ID__ - Unique number for each athlete
# - __Name__ - Athlete's name
# - __Sex__ - M or F
# - __Age__ - Integer
# - __Height__ - In centimeters
# - __Weight__ - In kilograms
# - __Team__ - Team name
# - __NOC__ - National Olympic Committee 3-letter code
# - __Games__ - Year and season
# - __Year__ - Integer
# - __Season__ - Summer or Winter
# - __City__ - Host city
# - __Sport__ - Sport
# - __Event__ - Event
# - __Medal__ - Gold, Silver, Bronze, or NA

# ## <a id="pie">1. Pie</a>  

# Let's draw the simplest graph possible. Display the percentages of three types of medals among the total number of medals.

# In[ ]:


colors = ['#f4cb42', '#cd7f32', '#a1a8b5'] #gold,bronze,silver
medal_counts = data.Medal.value_counts(sort=True)
labels = medal_counts.index
values = medal_counts.values

pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors))
layout = go.Layout(title='Medal distribution')
fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)


# All the main classes for drawing graphs are located in <strong>plotly.graph_objs</strong> as <strong>go</strong>:
# 
# - __go.Pie__ is a graph object with any of the named arguments or attributes listed below. 
# - __go.Layout__ allows you to customize axis labels, titles, fonts, sizes, margins, colors, and more to define the appearance of the chart.
# - __go.Figure__ just creates the final object to be plotted, and simply just creates a dictionary-like object that contains both the data object and the layout object.

# Okay, that was too easy. Let's complicate the chart a bit, we will use two Pie on one chart.
# 
# We will display the top 10 countries whose athletes win any medals. Separate for men and women.

# In[ ]:


topn = 10
male = data[data.Sex=='M']
female = data[data.Sex=='F']
count_male = male.dropna().NOC.value_counts()[:topn].reset_index()
count_female = female.dropna().NOC.value_counts()[:topn].reset_index()

pie_men = go.Pie(labels=count_male['index'],values=count_male.NOC,name="Men",hole=0.4,domain={'x': [0,0.46]})
pie_women = go.Pie(labels=count_female['index'],values=count_female.NOC,name="Women",hole=0.4,domain={'x': [0.5,1]})

layout = dict(title = 'Top-10 countries with medals by sex', font=dict(size=15), legend=dict(orientation="h"),
              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),
                             dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])

fig = dict(data=[pie_men, pie_women], layout=layout)
py.iplot(fig)


# - Parameter __hole__ sets the size of the hole in the center of the pie
# - Parameter __domain__ sets the offset. The X array set the horizontal position whilst the Y array sets the vertical. For example, x: [0,0.5], y: [0, 0.5] would mean the bottom left position of the plot.
# - Dict __annotations__ sets the format of the text inside the Pie.
# - To learn more, read the <a href='https://plot.ly/python/pie-charts/'>go.Pie documentation</a>

# ## <a id="bar">2. Bar</a>  

# Of course, we can not do without the Bar. 
# 
# Let's draw a bar chart of the number of sports in different years.

# In[ ]:


games = data[data.Season=='Summer'].Games.unique()
games.sort()
sport_counts = np.array([data[data.Games==game].groupby("Sport").size().shape[0] for game in games])
bar = go.Bar(x=games, y=sport_counts, marker=dict(color=sport_counts, colorscale='Reds', showscale=True))
layout = go.Layout(title='Number of sports in the summer Olympics by year')
fig = go.Figure(data=[bar], layout=layout)
py.iplot(fig)


# The whole rendering scheme is the same, now the base class is __go.Bar__.
# 
# - Dictionary __marker__ sets the drawing style of the chart and allows you to display the color scale
# - To learn more, read the <a href='https://plot.ly/python/bar-charts/'>go.Bar documentation</a>

# Again, let's complicate the graph and display the number of different medals for the top 10 countries

# In[ ]:


topn = 10
top10 = data.dropna().NOC.value_counts()[:topn]

gold = data[data.Medal=='Gold'].NOC.value_counts()
gold = gold[top10.index]
silver = data[data.Medal=='Silver'].NOC.value_counts()
silver = silver[top10.index]
bronze = data[data.Medal=='Bronze'].NOC.value_counts()
bronze = bronze[top10.index]

bar_gold = go.Bar(x=gold.index, y=gold, name = 'Gold', marker=dict(color = '#f4cb42'))
bar_silver = go.Bar(x=silver.index, y=silver, name = 'Silver', marker=dict(color = '#a1a8b5'))
bar_bronze = go.Bar(x=bronze.index, y=bronze, name = 'Bronze', marker=dict(color = '#cd7f32'))

layout = go.Layout(title='Top-10 countries with medals', yaxis = dict(title = 'Count of medals'))

fig = go.Figure(data=[bar_gold, bar_silver, bar_bronze], layout=layout)
py.iplot(fig)


# ## <a id="scatter">3. Scatter</a>  

# Let's draw a beautiful scatter plot showing average height and weight for athletes from different sports.
# 
# We will make circles of different sizes depending on the popularity of the sport and, as a result, the sample size of athletes.

# In[ ]:


tmp = data.groupby(['Sport'])['Height', 'Weight'].agg('mean').dropna()
df1 = pd.DataFrame(tmp).reset_index()
tmp = data.groupby(['Sport'])['ID'].count()
df2 = pd.DataFrame(tmp).reset_index()
dataset = df1.merge(df2) #DataFrame with columns 'Sport', 'Height', 'Weight', 'ID'

scatterplots = list()
for sport in dataset['Sport']:
    df = dataset[dataset['Sport']==sport]
    trace = go.Scatter(
        x = df['Height'],
        y = df['Weight'],
        name = sport,
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=10,
            size=df['ID'])
    )
    scatterplots.append(trace)
                         
layout = go.Layout(title='Mean height and weight by sport', 
                   xaxis=dict(title='Height, cm'), 
                   yaxis=dict(title='Weight, kg'),
                   showlegend=True)

fig = dict(data = scatterplots, layout = layout)
py.iplot(fig)


# It was beautiful, wasn't it? We can interactively remove the sport we are interested in, zoom in and analyze the charts in every possible way.
# 
# - Dictionary __marker__ again defines the drawing view, sets the shape type (try, for example, square), dimensions, and more. The possibilities are almost endless.
# - To learn more, read the <a href='https://plot.ly/python/line-and-scatter/'>go.Scatter documentation</a>

# ## <a id="box">4. Box</a> 

# We will display statistics on age for men and women participating in the Olympics with Boxplot.

# In[ ]:


men = data[data.Sex=='M'].Age
women = data[data.Sex=='F'].Age

box_m = go.Box(x=men, name="Male", fillcolor='navy')
box_w = go.Box(x=women, name="Female", fillcolor='lime')
layout = go.Layout(title='Age by sex')
fig = go.Figure(data=[box_m, box_w], layout=layout)
py.iplot(fig)


# - This graph describes the distribution of the data. The center vertical line corresponds to the median, and the boundaries of the rectangle correspond to the first and third quartiles. The points show the outliers. In addition, you can see the minimum and maximum values.
# - Who do you think is the __youngest__ (10 y.o.) and the __oldest__ (97 y.o.) participant of the Olympics? Find them :)
# - To learn more, read the <a href='https://plot.ly/python/box-plots/'>go.Box documentation</a>

# ## <a id="choropleth">5. Choropleth</a> 

# Let's determine how many participants were sent by different countries during the whole period of the Olympics.

# In[ ]:


#!pip install pycountry


# In[ ]:


def get_name(code):
    '''
    Translate code to name of the country
    '''
    try:
        name = pycountry.countries.get(alpha_3=code).name
    except:
        name=code
    return name

country_number = pd.DataFrame(data.NOC.value_counts())
country_number['country'] = country_number.index
country_number.columns = ['number', 'country']
country_number.reset_index().drop(columns=['index'], inplace=True)
country_number['country'] = country_number['country'].apply(lambda c: get_name(c))
country_number.head(3)


# In[ ]:


worldmap = [dict(type = 'choropleth', locations = country_number['country'], locationmode = 'country names',
                 z = country_number['number'], autocolorscale = True, reversescale = False, 
                 marker = dict(line = dict(color = 'rgb(180,180,180)', width = 0.5)), 
                 colorbar = dict(autotick = False, title = 'Number of athletes'))]

layout = dict(title = 'The Nationality of Athletes', geo = dict(showframe = False, showcoastlines = True, 
                                                                projection = dict(type = 'Mercator')))

fig = dict(data=worldmap, layout=layout)
py.iplot(fig, validate=False)


# - To learn more, read the <a href='https://plot.ly/python/choropleth-maps/'>go.Box documentation</a>

# That's it, you've learned how plotly works and mastered simple but beautiful interactive graphics.

# #### Useful links
# 
# - <a href='https://plot.ly/'>Plotly website</a>
# - <a href='https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners'>Plotly for beginners</a>
# - <a href='https://www.kaggle.com/hakkisimsek/plotly-tutorial-3'>Plotly tutorials</a>

# In[ ]:




