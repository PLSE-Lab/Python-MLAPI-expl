#!/usr/bin/env python
# coding: utf-8

# > Please wait for the graph loading (altair rendering..)
# 
# ![Chan Sung Jung](http://dmxg5wxfqgb4u.cloudfront.net/styles/card/s3/2018-11/zombie_stats.jpg?Ehv6JbNXuBv72ahO6z8vfBhq6olQxzfy&h=e449c3f0&itok=ACxa5AwR)
# 
# There are many **Interactive Visualization Tool**. such as :
# 
# - plotly / plotly.express
# - bokeh
# - altair
# 
# This kernel, I will use altair :) 
# 
# Let us all try to make a beautiful visualization.
# 
# ---
# 
# ### Table of Contents
# 
# 1. **Default Setting**
#     - 1.1. Import Library
#     - 1.2. Read Data & Columns Check
# 2. **Play Info Only (Altair for Starter)**
#     - 2-1. Play Info Type
#     - 2-2. # of Matches (Count Plot, Area Chart)
#     - 2-3. Weight Class (Barh Plot)
# 3. **Attack Analysis : RED vs BLUE**
#     - 3-1. Body (Scatter Plot)
#     - 3-2. Head (Scatter Plot + Axis)
#     - 3-3. Who won more? (5 types of Area Chart)
#         - Stacked Area Chart
#         - Normalized Stacked Area Chart
#         - Streamgraph
#         - Draw plots in row or col
#         - Draw with no stack

# ## 1. Default Setting
# 
# ### 1-1. Import Library

# In[ ]:


# default
import numpy as np 
import pandas as pd

# visualization
import missingno as msno
import altair as alt

# util
import os
import warnings
warnings.filterwarnings("ignore")


# You should have written very complex render code before, but now you can render with this code.
# 
# `alt.renderers.enable('kaggle')`
# 
# (The person who created this code will be blessed.)

# In[ ]:


alt.renderers.enable('kaggle')


# ### 1-2. Read Data & Columns Check
# 
# Let's look at the data first before the full-scale visualization.
# 
# And I will use `data.csv` file only.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'PATH = "/kaggle/input/ufcdata/"\ndata = pd.read_csv(PATH+\'data.csv\')')


# In[ ]:


print(data.shape)


# 145 columns?? There are quite a few columns.
# 
# First, let's list the names of the columns as a whole, and then select only the parts you need.

# In[ ]:


pd.options.display.max_columns = None # for see all columns
data.head(3)


# In[ ]:


col = data.columns
RedCol, BlueCol, PlayCol = [], [], []

for i in col:
    if 'R_' == i[:2]: RedCol.append(i)
    elif 'B_' in i[:2]: BlueCol.append(i)
    else : PlayCol.append(i)


# In[ ]:


print(f"Red Fighter Info : {len(RedCol)}\nBlue Fighter Info : {len(BlueCol)}\nPlay Info : {len(PlayCol)}")


# ## 2. Play Info Only
# 
# First, let's look at the information related to the game before analyzing both players.
# 
# ### 2-1. Play Info Type
# 
# Visualization by feature is as follows. (Just Simple Thinking)
# 
# - **Referee**
#     - Who did the most moderators? (Count Plot)
# - **Date** 
#     - Time Series Everything : Animation, Timely comparison
#     - Number of matches
# - **Location** 
#     - Where do some fighters win a lot? 
#     - Monthly Matches Animation
# - **Winner & weight_class** :
#     - Most Winner by Weight Class (Countplot)
# 
# Here I will draw only interesting plots. (in my opinion)

# In[ ]:


play_info = data[PlayCol]

play_info.head()


# ### 2-2. # of Matches
# 
# The data has been accumulating for quite a long time, so the days are very different.
# So this time, I did a visualization by year.
# 
# Below is the preprocessing.

# In[ ]:


play_info['year'] = play_info['date'].apply(lambda x : x.split('-')[0])
play_counts = pd.DataFrame(play_info['year'].value_counts().sort_index())
play_counts['count'] = play_counts['year']
play_counts['year'] = play_counts.index


play_counts.head()


# If you have a one-to-one matching column, visualization is straightforward.

# In[ ]:


alt.Chart(play_counts).mark_area(
    color="lightblue",
).encode(
    x='year',
    y='count'
)


# You can get a feel for it by looking at the code, but the basic form of altair is:
# 
# - `alt.Chart` : Make chart object
# - `mark_area` : In this part, call the desired graph type method.
#     - ex) mark_point(),  make_bar()
#     - Put the desired argument inside the method.
# - `encode` : Just put the elements you need on each axis.
# 
# 
# Here's how to draw a histogram + outline for a more visual effect:

# In[ ]:


alt.Chart(play_counts).mark_area(
    color="lightblue",
    interpolate='step-after',
    line=True
).encode(
    x='year',
    y='count'
)


# In this way, the properties can be set and studied as needed.
# 
# Anyway, you can see that the popularity jumped from 2005 to 2006.

# ### 2-3. Weight Class
# 
# Much simpler than the graph above. Let's use pandas to preprocess it and visualize it.

# First, let's see how many games are played in each weight class.
# 
# In this time, I use `make_bar` method.
# 
# You can decide the direction of the graph based on the x and y values.

# In[ ]:


weight_class_count = pd.DataFrame(data['weight_class'].value_counts())
weight_class_count['class'] = weight_class_count.index

alt.Chart(weight_class_count).mark_bar(
    color="#564d8d"
).encode(
    y='class',
    x='weight_class'
)


# The objects can then be combined and saved as follows (Here I created text with `mark_text`):
# 

# In[ ]:


class_bar = alt.Chart(weight_class_count).mark_bar(
    color="#564d8d"
).encode(
    y='class',
    x='weight_class'
)

class_text = class_bar.mark_text(
    align='left',
    baseline='middle',
    dx=3
).encode(
    text='weight_class'
)

(class_bar + class_text)


# Curiously, you can use simple expressions inside the encoding.
# 
# Let's draw the average line with `make_rule`.

# In[ ]:


class_bar = alt.Chart(weight_class_count).mark_bar(
    color="#564d8d"
).encode(
    x='class',
    y='weight_class'
)

rule = alt.Chart(weight_class_count).mark_rule(color='red').encode(
    y='mean(weight_class)'
)

(class_bar + rule).properties(width=600)


# You can also modify additional properties with `properties` in addition to the values.

# ## 3. RED vs BLUE
# 
# There are 69 features about Red Fighters & Blue Fighters
# 
# I want to visualize about attempts. How many attemps feature?

# In[ ]:


print(len([i for i in RedCol if 'att' in i]))
print([i for i in RedCol if 'att' in i])


# ### 3-1. Body

# First of all, check **no. of significant strikes to the body 'landed of attempted'**

# In[ ]:


# Unfortunately, Altair can visualize max 5000 points
# I will visualize latest 5000 matches
alt.Chart(data[-5000:]).mark_circle(size=10).encode(
    x='R_avg_BODY_att',
    y='B_avg_BODY_att',
    color='Winner',
).properties(
    width=500, 
    height=500,
    title='Average Body Attack'
).interactive()


# - You can make scatter plot by using `mark_circle`.
# - Add Color properties in Encode part
# - `interactive` method can make plot interactive.

# ### 3-2.  Head
# 
# Second, check **no. of significant strikes to the head 'landed of attempted'**
# 
# Now let's draw a more detailed scatter plot.

# In[ ]:


brush = alt.selection(type='interval')
base = alt.Chart(data[-5000:]).add_selection(brush)

points = base.mark_point(opacity=0.8).encode(
    x='R_avg_HEAD_att',
    y='B_avg_HEAD_att',
    color='Winner',
).properties(
    width=500, 
    height=500,
    title='Average Head Attack'
)

# Configure the ticks
tick_axis = alt.Axis(labels=False, domain=False, ticks=False)

x_ticks = base.mark_tick().encode(
    alt.X('R_avg_HEAD_att', axis=tick_axis),
    alt.Y('Winner', title='', axis=tick_axis),
    color=alt.condition(brush, 'Winner', alt.value('lightgrey'))
).properties(
    width=500, 
)

y_ticks = base.mark_tick().encode(
    alt.X('Winner', title='', axis=tick_axis),
    alt.Y('B_avg_HEAD_att', axis=tick_axis),
    color=alt.condition(brush, 'Winner', alt.value('lightgrey'))
).properties(
    height=500
)

# Build the chart
y_ticks | (points & x_ticks )


# There are a few things that are interesting.
# 
# - `alt.selection` means 'make named selection'
# - Inside the encoding, you can set the desired part with the `alt` method. (as `alt.condition`)
# - Use bitwise operators. ( | means horizonal add, & means vertical )
# - Graph objects are drawn in the order in which they are written. (yticks -> scatter -> xticks )
# - `mark_tick` can make tick(axis)
# 
# 
# By the way, Red certainly did well in the last 2000 games.
# 
# ### 3-3. Who won more? (Red or Blue)
# 
# There seems to be no correlation, but the previous game shows that the Reds won more.
# 
# This may be related to the athlete's morale. Or let me know in the comments if there are any rules.
# 
# there are many types to draw area chart
# 
# 1. Stacked Area Chart
# 2. Normalized Stacked Area Chart
# 3. Streamgraph
# 4. Draw plots in row or col
# 5. Draw with no stack

# In[ ]:


play_winner = play_info.groupby('year')['Winner'].value_counts().reset_index(name='counts')

# Check for Data Shape
display(play_winner.head())

# 1. Stacked Area Chart

alt.Chart(play_winner).mark_area().encode(
    x='year',
    y='counts',
    color='Winner'
).properties(
    width=800,
    title='Red or Blue, Who Win the GAME?'
)


# The first is a basic stacked area chart.
# You can see that the area chart is stacked.
# 
# The unfortunate part is that it's hard to know the ratio for the year.
# Normalized to this is shown below.

# In[ ]:


# 2. Normalized Stacked Area Chart

alt.Chart(play_winner).mark_area().encode(
    x='year',
    y=alt.Y("counts", stack="normalize"),
    color='Winner'
).properties(
    width=600,
    title='Red or Blue, Who Win the GAME?'
)


# If you want a more visual effect, streamgraph is also good.
# 
# It doesn't mean much, but it's great to show. This can help with scale comparison.
# 
# The more classes, the more colorful the images.

# In[ ]:


# 3. Streamgraph

alt.Chart(play_winner).mark_area().encode(
    alt.X('year:T',axis=alt.Axis(format='%Y', domain=False, tickSize=0)),
    alt.Y('counts:Q', stack='center', axis=None),
    alt.Color('Winner')
).properties(
    width=600
)


# Or you can draw them separately as rows and columns.

# In[ ]:


# 4. Draw in col or row

alt.Chart(play_winner).mark_area().encode(
    x='year',
    y='counts',
    color='Winner',
    column='Winner' # you can change this as row
)


# However, accurate size comparisons must be made on the same axis, on the same basis.
# 
# In this case, instead of stacking the graphs, you can draw them with higher transparency.

# In[ ]:


# 5. No Stacked & Set Opacity

alt.Chart(play_winner).mark_area(opacity=0.5).encode(
    x='year',
    y=alt.Y('counts:Q', stack=None),
    color='Winner',
).properties(
    width=600
)

