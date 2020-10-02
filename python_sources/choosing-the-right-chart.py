#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Data visualization
# There are so many charts out there, and choosing the appropriate one to explain a data best has always been a problem. So here is a compiled list based on some research.    
# This article is just about choosing the right charts. There is no code here.

# <a id=0></a>
# # Table of contents
# 1. [Comparing data](#1)
#     1. Bar Chart
#     1. Column Chart
#     1. Stacked Chart
#     1. Radar Chart
# 2. [Composition and Components](#2)
#     1. Pie Chart
#     1. Doughnut Chart
#     1. Pyramid Chart
#     1. Treemap Chart
#     1. Funnel Chart
# 3. [Tracking data over time](#3)
#     1. Line Chart
#     1. Spline Chart
#     1. Area Chart
#     1. Candlestick Chart
#     1. OHLC Chart
#     1. Sparkline Chart
# 4. [Analyzing distribution](#4)
#     1. Scatter Chart
#     1. Bubble Chart
#     1. Box Chart
#     1. Error Chart
#     1. Heatmap
#     1. Range Chart
#     1. Polar Chart
# 5. [Gauging Performance](#5)
#     1. Circular Gauge
#     1. Linear Gauge
#     1. Bullet Chart
# 6. [Project Data](#6)
#     1. Gantt Chart
# 7. [Geographical Data](#7)
#     1. Choropleth Map
#     1. Dot Map
#     1. Bubble Map
#     1. Flow Map

# <a id=1></a>
# # Comparing data
# ## Bar Chart and Column Chart
# These are used to compare quantitative values.     
# **Column Chart**
# <img style="max-height: 300px" src="https://ucarecdn.com/8e5016ad-5869-4469-904f-5b8d38fb29b3/samplesingleseriescolumnchart.png"/>
# **Bar Chart**
# <img style="max-height: 300px" src="https://ucarecdn.com/9add7407-7220-4c69-912f-08083279ef5c/barintro.png"/>
# 
# It is useful when there is only one variable, e.g. above we are comparing **sales** of each employee.
# Note, the only difference between a bar chart and column chart is that bar charts are horizontal while column charts are vertical.
# They do serve the same purpose and are interchangeable.  
# 
# ## Stacked Chart
# This is used similarly as bar charts, but it also focuses on the data and its segments.  
# <img style="max-height: 300px" src="https://ucarecdn.com/67aca98a-3cb2-43f5-838a-eef35b281d23/5stackedbarchartlargeopt.png"/>
# <img style="max-height: 300px" src="https://ucarecdn.com/ebb98d70-1a83-4cdf-bda4-c6be83baba42/7combinedmultiseriesbarandlinechartlargeopt.png"/>
# 
# Here, we are seeing how well each strategy works by looking at the total sales, as well as each product(segments).    
# We can stack bars on top of each other, or beside each other.  
# 
# 
# ## Radar Chart
# This is used for multivariate (two or more variables) data.
# ![image.png](https://ucarecdn.com/e9733021-344a-4aa9-989c-ed24cb19f523/Radar_Chart_01.png)
# Here, we are comparing **sales**, **marketing**, **development**, **customer support**, **adminstration** and **IT**.
# If it were just sales, we could use a bar/column chart. We could also stack all these variables on top of each other in a stacked bar chart if we are to treat them as part of a whole.
# 
# [Top](#0)

# <hr/>

# <a id=2></a>
# # Composition and Components
# This is comparing a part of the data to the whole.
# ## Pie chart
# This assumes the total is 100%, and it simply compares their shares.   
# <img style="max-height: 300px" src="https://ucarecdn.com/a74b783b-3582-45e6-9451-7b02834f3118/DjmXvGoUcAAVbzj.jpg"/>  
# One of the main criticism of pie charts is that it focuses on the area, and makes it difficult to read (especially when there are many data), and also when comparing multiple pie charts. As a rule of thumb, it is usually better to compare data by using a bar chart.  
# 
# In the example above, **women**, **1st class** and **2nd class** looks very similar in size, and without the percentage explicitly written, it would difficult to tell the difference.  
# 
# ## Doughnut chart
# <img style="max-height: 300px" src="https://ucarecdn.com/d365b4e7-0e08-49f6-b387-3a26e37dde3b/donut.png"/>  
# This is a pie chart with the centre cut out, but it has many advantages over a pie chart for the following reasons  
# * It is more accurate depiction because it focuses on the arc and circle circumference
# * It is a cleaner data and more readable
# * It is more space efficient as you can use the centre for adding more information
# 
# 
# ## Pyramid chart
# In order to display a simple hierarchy, use a pyramid chart. It stacks data on top of each other, from least to most or vice versa.  
# <img style="max-height: 300px" src="https://ucarecdn.com/ed6d97fa-ad4e-4fe1-9e14-681281ca555e/javascriptpyramidchart.png"/>
# 
# ## Treemap chart
# It compares a lot of data by using nested rectangles.     
# <img style="max-height: 300px" src="https://ucarecdn.com/0e4c8920-8fa7-4cc1-b2e4-13bbfebea925/treemap.png"/>  
# The sizes of each rectangle indicate the relative size of each data, hence it's easy to spot patterns in treemap charts.      
# 
# 
# ## Funnel chart
# It is used to compare the different stages, usually sales and identify the numbers.   
# <img style="max-height: 300px" src="https://ucarecdn.com/45dc376d-783f-4a52-9caf-f38f5cceed44/funnel_chart_1.jpg"/>   
# It shows an increasing amount where the initial stage is always bigger than the next stage, and the first stage will be 100%.  
# It is like stacking bar charts on top of each other.   
# For example, to compare the number of people who viewed the apps, the number of people who downloaded it and finally the number of people who are using it.      
# As shown above, we start with the number of conversation with sales reps(100%), to finally the total number of sales.   
# 
# [Top](#0)

# <hr/>

# <a id=3></a>
# # Tracking data over time
# These are charts used to track differing sizes or changes over time
# ## Line Chart
# It is used to show trends, e.g. stock prices.    
# <img style="max-height: 300px" src="https://ucarecdn.com/d1996f5b-af1d-4dee-9a5a-d7ef79c46026/labelsatend1024x650.jpg"/>  
# If one is to compare a **single data over time** (e.g. sales in January, February and March), or **multiple data** at **a single time**(e.g. orange, apple and banana sales in January) you can use a bar chart. However, in order to compare multiple data over a duration, we will need to use **line chart**.
# 
# 
# ## Spline Chart
# Basically line chart with points, perhaps to highlight somethings like the peak, nadir, important date etc.
# <img style="max-height: 300px" src="https://ucarecdn.com/79ad24f6-9933-4472-8f9b-2a2fd790ce37/spline_charts.gif"/>
# 
# ## Area Chart
# It is based on line-charts; in fact, line charts can be converted into area charts and vice versa.
# Overlapping data are indicated by using adjusting transparency.    
# ### Choosing between Line and Area chart.
# 1. When there are **more than two** overlapping data, you should use line chart. The more the data, the harder it is to show in area chart.
# <img style="max-height: 300px" src="https://ucarecdn.com/a6c4a5cf-d2a7-4bfb-b066-189bdd4138a1/US_and_USSR_nuclear_stockpilessvg.png"/>
# 2. When you have a **part to whole data**, you should use area chart.
# <img style="max-height: 300px" src="https://ucarecdn.com/a12bfb09-6340-47ff-8ec1-b6265bf71f81/areacharthotelssample.png"/>
# 3. You can also used stacked Area chart for multiple data to **compare ratio** and track **total amount**.
# <img style="max-height: 300px" src="https://ucarecdn.com/cfdcb47a-d5b9-48d6-952e-5ddcac800f7a/areachart.png"/>
# 
# ## Candlestick chart (OHLC)
# Also called Japanese candlestick chart, is a financial chart used to show changes in stuff like securities, stock, currency etc.   
# <img style="max-height: 300px" src="https://ucarecdn.com/0c3bf954-52a1-4fdb-b5df-f28d3c06b605/0YaVHdOICt_PTqlhw.png"/>  
# It is a combination of line and bar charts.    
# OHLC stands for Open High Low Close. 
# Open and Close indicate the opening and close prices for the day, respectively.
# High and Low represents the highest and the lowest price for that day.
# <img style="max-height: 300px" src="https://ucarecdn.com/0a1eb6ff-caf3-4e98-ba47-ef2d97f4091e/CandlestickBasicsChart.gif"/>
# In stock, bullish indicates it is rising while bearish indicates the price is dropping.     
# 
# ## Sparkline
# It is a small line/area chart without axes. It is used to show a quick view of the variation and trend.
# <img style="max-height: 300px" src="https://ucarecdn.com/81d4e2e3-f077-417b-ba3f-ab7e904e20a3/sparklinechart.png"/>
# 
# [To

# <hr/>

# <a id=4></a>
# # Analyzing distribution
# ## Scatter Chart
# Also known as scatter plot.  
# It is used to get correlation between two data.
# <img style="max-height: 300px" src="https://ucarecdn.com/a0e24ce8-1f72-4144-bb43-69cfaa156b60/ScatterPlotsCorrelationExample.jpg"/>
# <img style="max-height: 300px" src="https://ucarecdn.com/6def002e-0c7a-4d15-a1ba-db33659cda38/ScatterDiagramScatterPlotScatterGraph.jpg"/>
# ## Bubble Chart
# It's used to add a $3^{rd}$ dimension to a scatter plot by using proportionate circles instead of dots.   
# <img style="max-height: 300px" src="https://ucarecdn.com/cb39f245-8ab1-4b16-87d0-ae467d2453bc/bubblecharttemplate.png"/>  
# It's is used for multivariate data and interacting with bubble chart may be used to provide more information.
# For example, we can plot age vs average spending, and we can use the area of the circle to determine the relative population sizes of each age group.   
# Here, the area of the circle indicates the likelihood of success.
# 
# ## Box Chart
# It's used to display ranges(quartiles) and median values.
# <img style="max-height: 300px" src="https://ucarecdn.com/185e3697-7d97-4838-b416-24c31af20020/BoxWhisker1.png"/>
# <img style="max-height: 300px" src="https://ucarecdn.com/489e130c-8a0a-4ca1-8481-a5b9dd0cd670/1Dqj4.jpg"/>
# 
# ## Error Chart
# You may add it to line chart, bar chart, scatter chart etc to indicate degree of certainty. For example, the heart rate of people in 20s have heart rate of 71$\pm$3
# <img style="max-height: 300px" src="https://ucarecdn.com/a6de2e0e-d237-49a9-a546-09cc513265a6/phperrorchartsgraphs.png"/>
# <img style="max-height: 300px" src="https://ucarecdn.com/2c4b9961-da6d-477e-93d0-b93be35f7a97/phperrorlinechartsgraphs.png"/>
# 
# ## Heatmap
# <img style="max-height: 300px" src="https://ucarecdn.com/2a80e0d8-0f72-415d-9ef2-196a96a7149e/heatmap1140.jpg"/>  
# It is used to compare two large or complex dataset and make it very easy to visualizes their relationships. You can use it as a gradient to indicate things like severity, correlation, concentration etc.
# 
# 
# ## Range (Diverging) Chart
# It is used when the ranges are important, and you want to focus on the outermost values. It's basically stacking of bar charts
# <img style="max-height: 300px" src="https://ucarecdn.com/372f1dca-5061-4359-8055-f732459ffeb3/jsprangebarchartsgraphs.png"/>
# <img style="max-height: 300px" src="https://ucarecdn.com/8a7f27db-eda2-4d41-8355-de81a27fc172/rangechartsdivergingbarchart.png"/>
# 
# ## Polar Chart
# <img style="max-height: 300px" src="https://ucarecdn.com/c79a0e3b-f366-492e-8883-b808c13c7d11/stackrose_g.png"/>  
# Also called: Coxcomb chart, Rose chart, Polar Area Chart.    
# The Polar Area chart is similar to the usual pie chart. It measures data based on angles and how long it extends from the centre. It is for data which are best represented by angles.
# This chart above is used to show both the wind direction and speed accurately.
# 
# [Top](#0)

# <hr/>

# <a id=5></a>
# # Gauging Performance
# ## Linear and Circular gauge
# <img style="max-height: 300px" src="https://ucarecdn.com/398d10db-0a18-47d0-8ac1-3578851d3910/thermometerstylechartexcel2010ingauges.jpg"/>  
# This is used when a data indicates something like speed, height, electricity, temperature etc.
# Thermometer and Odometer are excellent examples of **Linear** and **Circular gauge** respectively.
# 
# 
# ## Bullet Chart
# This is a variation of a bar graph, but it contains more information like
# <img style="max-height: 300px" src="https://ucarecdn.com/52fef9e8-811a-45cd-911c-29d2cb94ee03/Bullet_graphs_multiple.png"/>  
# 1. The background colour indicates severity or some range. Usually, this can be differentiated by the shades of the colour.
# 1. The centre line indicates the actual value.
# 1. The short dark mark indicates the target or ideal value.
# 
# Like any bar char, it could be horizontal, vertical or stacked.
# 
# According to the graph above, regarding **revenues**.
# 1. The ranges are 0-150, 150-225 and 225 to 300
# 1. The actual value is around 275
# 1. The target 250
# 
# [Top](#0)

# <hr/>

# <a id=6></a>
#  # Project data
#  ## Gantt Chart
#  This is used to monitor progress and illustrate the schedule of a project. It's a variation of bar chart.
# <img style="max-height: 300px" src="https://ucarecdn.com/bc658f18-2ad1-41e5-915b-21713d7af630/ScreenShot20170331at81817AM640x382.png"/>
#  
#  [Top](#0)

# <hr/>

# <a id=7></a>
# # Geographical Map
# ## Choropleth map
# This is basically an heat graph on a geographical map. It uses different shading or colouring within a predefined areas to indicate the average values of that particular area.
# <img style="max-height: 300px" src="https://ucarecdn.com/333b7b25-650a-499f-90f0-cbe65d68664d/africalifeexp.gifg"/>
# 
# ## Dot map
# It is used to indicate the concentration of a phenomenon or an occurrence. It may also show the presence or absence of such phenomena or features.    
# Unlike **choropleth**, it doesn't mind boundaries.
# <img style="max-height: 300px" src="https://ucarecdn.com/ec941dfd-d9a0-4345-a010-8c13bfa3ab81/DotMapsWorldPopulation.jpg"/>
# <img style="max-height: 300px" src="https://ucarecdn.com/91cd2e6a-c9b4-4cd0-9359-b8164f8d2b49/DOTDENSITY.gif"/>
# 
# ## Bubble map
# It is one of the best ways to communicate comparable location-based data clearly and concisely. A good example is the GDP of countries. 
# It's easy to compare sizes with bubble maps.
# <img style="max-height: 300px" src="https://ucarecdn.com/45a548fa-9587-41d4-a06c-09e8a6ac206d/EuropeGDPpercountry1.jpg"/>
# 
# ## Flow map
# This is the combination of geographical map and flow chart. It indicates the rate or size of transfer between two regions.
# <img style="max-height: 300px" src="https://ucarecdn.com/c4fda0cf-bdf9-438f-a71e-0d19c4879644/ustrademap.png"/>
# 
# [Top](#0)

# 
