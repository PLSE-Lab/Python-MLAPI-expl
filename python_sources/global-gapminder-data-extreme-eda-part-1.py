#!/usr/bin/env python
# coding: utf-8

# # Gapminder Dataset

# # This notebook is Part 1 of Processes performed over Gapminder dataset. Basically population distribution and trend has been observed.

# Importing Required Libraries 

# # NOTE: 
# plotly.plotly has been deprecated and be updated to chart_studio

# In[ ]:


import numpy as np      # for supporting large, multi-dimensional arrays and matrices
import pandas as pd     # for data manipulation and analysis 
import matplotlib.pyplot as plt 


# In[ ]:


import plotly.offline as py            # Plotly is a graphing python library that offers more than 40 charts
py.init_notebook_mode(connected=True)  # This method is called from the offline method that require plotly.js to be loaded into the notebook dom


# In[ ]:


import plotly.graph_objs as go         # This package imports definitions for all of Plotly's graph objects
# import chart_studio.grid_objs as go


# In[ ]:


import plotly.express as px                      # High-level API for rapid data exploration and figure generation


# # Loading data

# In[ ]:


from plotly.figure_factory import create_table   # figure_factory includes many wrapper functions that create unique chart types that are not yet included in plotly.js
gm = px.data.gapminder()                         # Loading data into gm, gm is a dataframe

table = create_table(gm.head(10))
table


# In[ ]:


type(gm) # gm is a dataframe as stated above


# # Exploratory Data Analysis

# In[ ]:


gm.shape


# In[ ]:


gm.columns


# In[ ]:


gm.info()


# In[ ]:


gm.describe()


# In[ ]:


from pandas_profiling import ProfileReport # open source python module that helps to return in depth EDA report with a couple of lines of code
report = ProfileReport(gm)
report


# In the Overview part of the report, it says that the number of missing values is zero. So our half of the work is done. Still showing other ways to check the missing values by python code as well as visualisation.

# In[ ]:


gm.isnull()


# In[ ]:


gm.isnull().sum()


# In[ ]:


import seaborn as sns
sns.heatmap(gm.isnull(), yticklabels = False, cmap = "Greens")


# It was already known that the dataset do not have any null value.... this is just an another way to check the null values as at times, with huge datasets, the columns in the middle having null values are not printed out. In such cases the visualisation used above proves to be really good.

# ========================================================================================================================

# Also from the report we get to know that there are three categorical data i.e. country, continent, iso_alpha. 

# Understanding the data in a better way

# In[ ]:


gm.columns


# In[ ]:


unique_continent = gm.continent.unique()
unique_continent = unique_continent.tolist()
unique_continent


# In[ ]:


type(unique_continent)


# Understanding the data in a better way using cross tabulation

# In[ ]:


gm_edit = gm.copy()


# In[ ]:


gm_edit.drop(['year', 'lifeExp', 'gdpPercap','iso_alpha', 'iso_num'], axis = 1, inplace = True)
gm_edit.head()


# In[ ]:


gm.pivot_table(index=['continent'], aggfunc='size')


# In[ ]:


ctr1 = gm['country'].loc[gm['continent'] == 'Africa'].unique()
print("Countries in Africa: ", ctr1)
print("\nTotal number of countries in Africa: ", len(ctr1))


# In[ ]:


ctr2 = gm['country'].loc[gm['continent'] == 'Americas'].unique()
print("Countries in Americas: ", ctr2)
print("\nTotal number of countries in Americas: ", len(ctr2))


# In[ ]:


ctr3 = gm['country'].loc[gm['continent'] == 'Asia'].unique()
print("Countries in Asia: ", ctr3)
print("\nTotal number of countries in Asia: ", len(ctr3))


# In[ ]:


ctr4 = gm['country'].loc[gm['continent'] == 'Europe'].unique()
print("Countries in Europe: ", ctr4)
print("\nTotal number of countries in Europe: ", len(ctr4))


# In[ ]:


ctr5 = gm['country'].loc[gm['continent'] == 'Oceania'].unique()
print("Countries in Oceania: ", ctr5)
print("\nTotal number of countries in Oceania: ", len(ctr5))


# To understand in depth information like population of a continent or a country in a particular year.... click on the respective tag
# For example. if you wish to know more about Asia, click on Asia.
# To visualize the plot in a better way, say you wish to know more about India, then click on India. This will return you all info of population in India. 
# Also you can hover the mouse pointer over any of the required field and you will get the information for the same.
# 
# To go back to the previous chart, click again on the same field you chose earlier. Like click again on India and then Asia and you are back to the original chart...

# In[ ]:


plt.figure(figsize = (15,10))
fig = px.sunburst(gm, path=['continent', 'country', 'year'], values='pop', color='continent',
                  color_discrete_map={'Asia':'blue', 'Europe':'green', 'Africa':'red', 'Americas':'orange', 'Oceania':'black'}, 
                  title = "Population Distribution")
fig.show()


# ========================================================================================================================

# Oobserving the trend in population change in the continents individually and the top two countries with highest population according to the data set!

# # Bar plot Visualisation

# The below projections are dynamic where you can 
# 
# 1) Download Plot as a png</br>
# 
# 2) Zoom</br>
# 
# 3) Pan </br>
# 
# 4) Box Select</br>
# 
# 5) Lasso Select</br>
# 
# 6) Zoom in </br>
# 
# 7) Zoom out</br>
# 
# 8) Autoscale</br>
# 
# 9) Reset Axis</br>
# 
# 10) Toggle Spike Lines</br>
# 
# 11) Show Closest Data on Hover</br>
# 
# 12) Compare data on hover</br>

# In[ ]:


# Hover over the bar to get the exact data
df_asia =gm.query("continent == 'Asia'")  # query() method queries the columns of a DataFrame with a boolean expression.
fig = px.bar(df_asia, x='year', y='pop', height=450, title = "Population of Asia", color = 'country')
fig.show()


# In the above plot, hover the mouse pointer over various colored blocks where each coloured block represent a country of Asia. Year by year the population of most of the countries have been observed to be increasing. 

# In[ ]:


# Hover over the bar to get the exact data
df_india = gm.query("country == 'India'")
fig = px.bar(df_india, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], height=450, title = "Population of India", color = 'country')
fig.show()


# In[ ]:


df_china = gm.query("country == 'China'")
fig = px.bar(df_china, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], color='pop',
             labels={'pop':'Population'}, height=500, title = "Population of China")
fig.show()


# In[ ]:


# Hover over the bar to get the exact data
df_americas =gm.query("continent == 'Americas'")
fig = px.bar(df_americas, x='year', y='pop', height=450, title = "Population of Americas", color = 'country')
fig.show()


# In[ ]:


# Hover over the bar to get the exact data
df_us = gm.query("country == 'United States'")
fig = px.bar(df_us, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], height=450, title = "Population of United States", color = 'country')
fig.show()


# In[ ]:


df_brazil = gm.query("country == 'Brazil'")
fig = px.bar(df_brazil, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], color='pop',
             labels={'pop':'Population'}, height=500, title = "Population of Brazil")
fig.show()


# In[ ]:


# Hover over the bar to get the exact data
df_africa =gm.query("continent == 'Africa'")
fig = px.bar(df_africa, x='year', y='pop', height=450, title = "Population of Africa", color = 'country')
fig.show()


# In[ ]:


# Hover over the bar to get the exact data
df_nigeria = gm.query("country == 'Nigeria'")
fig = px.bar(df_nigeria, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], height=450, title = "Population of Nigeria", color = 'country')
fig.show()


# In[ ]:


df_egypt = gm.query("country == 'Egypt'")
fig = px.bar(df_egypt, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], color='pop',
             labels={'pop':'Population'}, height=500, title = "Population of Egypt")
fig.show()


# In[ ]:


# Hover over the bar to get the exact data
df_europe =gm.query("continent == 'Europe'")
fig = px.bar(df_europe, x='year', y='pop', height=450, title = "Population of Europe", color = 'country')
fig.show()


# In[ ]:


# Hover over the bar to get the exact data
df_germany = gm.query("country == 'Germany'")
fig = px.bar(df_germany, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], height=450, title = "Population of Germany", color = 'country')
fig.show()


# In[ ]:


df_uk = gm.query("country == 'United Kingdom'")
fig = px.bar(df_uk, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], color='pop',
             labels={'pop':'Population'}, height=500, title = "Population of 'United Kingdom")
fig.show()


# On seeing the above two plots, it is observed that the population growth rate in the above two countries is quite stable and not increasing steeply. It seems like the number of birthd rate and death rate is approximately equal keeping the total population growth over years to be seemingly quite stagnant.

# In[ ]:


# Hover over the bar to get the exact data
df_oceania =gm.query("continent == 'Oceania'")
fig = px.bar(df_oceania, x='year', y='pop', height=450, title = "Population of Oceania", color = 'country')
fig.show()


# In[ ]:


# Hover over the bar to get the exact data
df_australia = gm.query("country == 'Australia'")
fig = px.bar(df_australia, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], height=450, title = "Population of Australia", color = 'country')
fig.show()


# In[ ]:


df_nz = gm.query("country == 'New Zealand'")
fig = px.bar(df_nz, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap', 'iso_alpha', 'iso_num'], color='pop',
             labels={'pop':'Population'}, height=500, title = "Population of 'New Zealand")
fig.show()


# In[ ]:


df_nz


# ==========================================================================================================================
# 

# In[ ]:


pip install bubbly


# In[ ]:


pip install chart-studio 


# The plotly.plotly module is deprecated, so for resolving it please install the chart-studio package and use the chart_studio.plotly module instead. or you can even install the latest version of plotly )

# In[ ]:


pip install plotly==4.6.0


# In[ ]:


from bubbly.bubbly import bubbleplot 
#from plotly.plotly import iplot  
from plotly.offline import iplot
#from chart_studio.plotly import  iplot

figure = bubbleplot(dataset=gm, x_column='gdpPercap', y_column='lifeExp', 
    bubble_column='country', time_column='year', size_column='pop', color_column='continent', 
    x_title="GDP per Capita", y_title="Life Expectancy", title='Global Statistical Indicators',
    x_logscale=True, scale_bubble=3, height=650)

iplot(figure, config={'scrollzoom': True})


# =======================================================================================================================

# # Part 2 - to be contd......
# Coming Soon
# 

# # The upcoming parts of this series will include more sophiticated and dynamic visualisations to play around with... 
# If you wish to receive the notification about the same, do follow me and if you liked the work then surely give an upvote. 
# 
# Also please give suggestions on what type of work do you wish me to make public and also suggestions on how to improve my work will be much appreciated...\\
# 

# In[ ]:




