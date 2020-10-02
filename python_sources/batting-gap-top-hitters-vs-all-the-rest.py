#!/usr/bin/env python
# coding: utf-8

# #In process....check back later as I continue to work on this project.  Thanks!
# #The Batting Gap
# 
# ##Top Hitters vs. Everyone
# I am newbie when it goes to data science and machine learning.  So I started to take some classes (the free ones of course).
# I started a class on Udacity that had a final project.  It was to pick a data set and done some data analyst create some visualizations and share it with everyone.  So this workbook is exploring baseball data specifically from 1969-2015 and specifically batting data.
# 
# As the title suggests I am doing my first analysis on batting data and as I started to work on my analysis I noticed some interesting things as it relates to the top hitters and everyone else.
# 
# Below are some finds that I hope you find interesting.  Please comment below in the comments if you have any input or future knowledge to what the data is telling us.

# In[ ]:


#Importing the packages I will need for analysis 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Using Bokeh for charting, I like the charts and the interaction they provide.
from bokeh.io import push_notebook
from bokeh.plotting import figure, show, output_notebook
import bokeh.charts as bcharts
import bokeh.charts.utils as bchartsU


# In[ ]:


#Focusing on just the batting, have some ideas on the project to add data from some other areas.
full_frame = pd.read_csv("../input/batting.csv")


# ##Filtering Out Data
# Wanted to filter out any data before 1969.  I had seen someone else doing some baseball analysis do this and after looking at the older data it seemed reasonable to do the same here.
# 
# Second part creates a new column with the calculation for batting average for each record.  Basically each year and player gets a batting average calculated.
# 

# In[ ]:


frame = full_frame[full_frame['year']>=1969]

frame = frame.assign(AVG = frame['h'] / frame['ab'])


# #Getting Data Ready for Plots
# First analysis I wanted to do was check out three key batting stats.  RBIs, Home Runs and Batting Average.  What I am doing below is grouping the data to get averages of these stats each year.  I am also creating a group of data that looks at just the top 50 players each year and taking their averages.  You will see why I did this later on.

# In[ ]:


frameYearTripleCrown = frame.groupby(['year'], as_index=False)[['rbi', 'hr', 'AVG']].mean()

frameLessCols_AVG = frame.loc[:, ['year', 'player_id', 'AVG']]
yearPlayerAVGSorted = frameLessCols_AVG.sort_values(by=['year', 'AVG'], ascending=[True, False])
topPlayersDetails_AVG = yearPlayerAVGSorted.groupby(['year'], as_index=False).head(50)
topPlayersAVG = topPlayersDetails_AVG.groupby(['year'], as_index=False)['AVG'].mean()


# In[ ]:


x = frameYearTripleCrown['year']
y = frameYearTripleCrown['rbi']
y2 = frameYearTripleCrown['hr'] 
y3 = frameYearTripleCrown['AVG']
y3_1 = topPlayersAVG['AVG']
output_notebook()


# In[ ]:


p = figure(title="Mean RBI by Year", plot_height=450, plot_width=800)
p2 = figure(title="Mean HR's by Year", plot_height=450, plot_width=800)
p3 = figure(title="Mean Batting Average by Year Top Players vs All Players", plot_height=450, plot_width=800)
p3_A = figure(title="Mean Batting Average by Year", plot_height=450, plot_width=800)

c = p.circle(x, y, radius=0.8, alpha=0.5)
c2 = p2.circle(x, y2, radius=0.8, alpha=0.5)
c3 = p3.circle(x, y3, radius=0.8, alpha=0.5)
c3_1 = p3.circle(x, y3_1, radius=0.8)
c3_A = p3_A.circle(x, y3, radius=0.8, alpha=0.5)


# #Output
# This first chart plots the averages over time.  As you can see we see we have some ups and downs.  I wanted to understand a little bit more about the changes so I decided to look at the top hitters.  What type of trend are we seeing around the best players when it comes to these stats.

# In[ ]:


show(p3_A)


# #Top Hitters vs Everyone
# Here we can see a little bit different of a curve.  It appears that top players while having some ups and downs as well have a little bit of an upward trend.  The chart even shows an upward trend starting in the 90's whereas up above we see a downward trend from the 80's until current.  It appears that while everyone else was seeing a dip in the average our big hitters were still swining away.

# In[ ]:


show(p3)

