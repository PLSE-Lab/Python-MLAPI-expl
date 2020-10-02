#!/usr/bin/env python
# coding: utf-8

# #INTRODUCTION
# 
# Bokeh is a data visualization library of python which provides high performance intractive charts and plots. 
# Here I am going o create a plot which will change based on the value selected from the drop down list.Here the dataset I have used is Gapminder data.
# I have divided this into 2 parts:
# 
# 1. Part 1 :      
#     * Data exploration with pandas
#     * Bokeh packages use
#     * Creating Drop Down menu
#     * Creating Plots 
# 
# 1. Part 2 :
#     * Linking Plots
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


gap = pd.read_csv('/kaggle/input/gapminder/gapminder_tidy.csv')


# In[ ]:


gap.info()


# In[ ]:


gap.head()


# Importing bokeh library to create our basic visualization.

# In[ ]:


#output_file-to save the layout in file, show-display the layout , output_notebook-to configure the default output state  to generate the output in jupytor notebook.
from bokeh.io import output_file, show, output_notebook 
#ColumnDataSource makes selection of the column easier and Select is used to create drop down 
from bokeh.models import ColumnDataSource, Select
#Figure objects have many glyph methods that can be used to draw vectorized graphical glyphs. example of glyphs-circle, line, scattter etc. 
from bokeh.plotting import figure 
#To create intractive plot we need this to add callback method.
from bokeh.models import CustomJS 
#This is for creating layout
from bokeh.layouts import column
output_notebook() #create default state to generate the output


# Here I am creating a plot which shows rate of change of fertility of each country over the year. Select any country and visualize the rate of that country. we can do the same for other columns ex- population, life. child_mortality and gdp. 
# I have created 2 dataframe from original dataframe, first for the overall data and second one is for the default selection option.
# 
# CustomJS is used for changing the plot based on selection.
# cb_obj.value will hold the current selection value and based on that I am filtering the data of source and updating the current data.

# In[ ]:


gap1=gap.loc[:, ['Country','Year', 'fertility']]
gap2 = gap1[gap1['Country'] == 'Iceland' ]
   

Overall = ColumnDataSource(data=gap1)
Curr=ColumnDataSource(data=gap2)


#plot and the menu is linked with each other by this callback function
callback = CustomJS(args=dict(source=Overall, sc=Curr), code="""
var f = cb_obj.value
sc.data['Year']=[]
sc.data['fertility']=[]
for(var i = 0; i <= source.get_length(); i++){
	if (source.data['Country'][i] == f){
		sc.data['Year'].push(source.data['Year'][i])
		sc.data['fertility'].push(source.data['fertility'][i])
	 }
}   
   
sc.change.emit();
""")
menu = Select(options=list(gap['Country'].unique()),value='Iceland', title = 'Country')  # drop down menu
p=figure(x_axis_label ='Year', y_axis_label = 'fertility') #creating figure object 
p.circle(x='Year', y='fertility', color='green', source=Curr) # plotting the data using glyph circle
menu.js_on_change('value', callback) # calling the function on change of selection
layout=column(menu, p) # creating the layout
show(layout) # displaying the layout


# # CONCLUSION
# 
# Working on Bokeh library is fun, isn't ?
# If you are also interested in learning Bokeh, I am going to dive deeper into this.
# 
# **If you have any query I am happy to hear it.**
