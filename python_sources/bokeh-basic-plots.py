#!/usr/bin/env python
# coding: utf-8

# # What is Bokeh
# Bokeh is an interactive visualization library that targets modern web browsers for presentation. It is good for:
# 
#     Standalone HTML documents, or server-backed apps
#     Expressive and versatile graphics
#     Large, dynamic or streaming data
#     Easy usage from python (or Scala, or R)
# 

# # NoJSRequired
# It uses javascript in background to build charts and figures. We don't have to focus on heavy js coding to make plots bokeh provides easy implementation of plots using python .We can create interactive plots, dashboards, and data applications very easily and quickly

# In[ ]:


from bokeh.io import output_notebook, show #,output_file 
output_notebook()


# Above import will let you visualize plots in notebook itself there is other option as well i.e making html page of plot , for that all you need to do is import output_file it will create a html file

# In[ ]:


import numpy as np
from bokeh.plotting import figure
#figure will create an empty plot where you can visualize data
#lets take a look at simple sin plot


# In[ ]:


#create values for plotting data
x = np.arange(0,10,0.5)
#sin values of x
y=np.sin(x)


# In[ ]:


#create empty figure having axis lables 
#having axis label with title of plot and height and width of 500
p = figure(x_axis_label='number',y_axis_label='sin(number)',title="Sin Plot", plot_width=500, plot_height=500)
#line is a glyph lets talk a little later for now you can see that it creates line plot
p.line(x,y)
#show will show rendered plot
show(p)


# From above plot we can see sin plot to right of plot we have few option which make bokeh interactive and dynamic options like
#          1. pan 
#          2. box zoom
#          3. wheel zoom
#          and so on
# Its is tools parameter in figure where we can specify which tool to embedded into plot

# # What are Glyphs?
# The basic visual building blocks of Bokeh plots, e.g. lines, rectangles, squares, wedges, patches, etc. The bokeh.plotting interface provides a convenient way to create plots centered around glyphs.
# 
# List of glyphs
#  
#     asterisk()
#     circle()
#     circle_cross()
#     circle_x()
#     cross()
#     diamond()
#     diamond_cross()
#     inverted_triangle()
#     square()
#     square_cross()
#     square_x()
#     triangle()
#     x()
# Feel free to explore each glyph

# In[ ]:


import pandas as pd 
from sklearn.datasets import load_iris # iris dataset


# In[ ]:


iris = load_iris()
data = iris.data
column_names = iris.feature_names
#Creating dataframe out of data and features
#convert target name from numeric to names
df = pd.DataFrame(data,columns=column_names)
df = df.assign(target=iris.target)
df.target = df.target.apply(lambda x: iris.target_names[x])


# In[ ]:


df.head()


# # ColumnDataSource
# 
# The ColumnDataSource is the core of most Bokeh plots, providing the data that is visualized by the glyphs of the plot.One of best feature of ColumnDataSource is that it makes it easy to have multiple plots share same data.
# 
# ColumnDataSource takes dataframe or groupby as parameter
# 
# Lets see a simple plot using ColumnDataSource

# In[ ]:


from bokeh.models import ColumnDataSource


# In[ ]:


#columndatasource takes dataframe in parameter
source = ColumnDataSource(data=df)
#plot data with glyph
p = figure(x_axis_label='sepal length (cm)', y_axis_label='petal length (cm)')
p.circle(x='sepal length (cm)', y='petal length (cm)', source=source)
show(p)


# In[ ]:


from bokeh.models import CategoricalColorMapper

#CategoricalColorMapper is used to map color to factors all we need to do is initialize CategoricalColorMapper
#with factor that you want to and mention palette i.e color
color_mapper = CategoricalColorMapper(factors=np.unique(df.target),
                                      palette=['red', 'green', 'blue'])
#add dictionary to circle with field and transformer 
p.circle(x='sepal length (cm)', y='petal length (cm)', source=source,
         color=dict(field='target',transform=color_mapper),
            legend='target')

show(p)


# As you can see that values and factors are closely related as low values corresponds to setosa .
#   
#  Bokeh has easy implementation for colors as well , you can use palette package to import different range of colors
#  
#  Lets see an example on how palette can be used in plots

# In[ ]:


from bokeh.palettes  import Spectral3
from bokeh.layouts import gridplot


# In[ ]:


#make a source for dataframe this was already initialized already in above part 
source = ColumnDataSource(data=df)

#create CategoricalColorMapper
color_mapper = CategoricalColorMapper(factors=np.unique(df.target),
                                      palette=Spectral3)
#tools 
TOOLS = "box_select,lasso_select,help"

# create a new plot and add a renderer
left = figure(tools=TOOLS, plot_width=400, plot_height=400, title=None)
left.circle(x='sepal length (cm)', y='petal length (cm)', source=source,
         color=dict(field='target',transform=color_mapper),legend='target')
left.legend.location='top_left'
# create another new plot and add a renderer
right = figure(tools=TOOLS, plot_width=400, plot_height=400, title=None)
right.circle(x='petal length (cm)', y='petal length (cm)', source=source,
         color=dict(field='target',transform=color_mapper),legend='target')
right.legend.location='top_left'

#Here we use gridplot for side by side plotting 
p = gridplot([[left, right]])

show(p)


# Box_select allows to select data in plot as boxes where as lasso_select make you selet region selected without any specific shape give it a try in above plot to see difference
# 
# The above plots are linked plot which makes bokeh special

# # Working with catgorical Data

# In[ ]:


new_df = df.sample(50)
#get count of target data 
group=new_df.groupby('target').count()
#change x axis  to factors 
p = figure(x_range=group.index.tolist(), plot_height=250, toolbar_location=None)
#apply vbar glyph for bar plot
p.vbar(x=group.index.tolist(), top=group['sepal length (cm)'].tolist(), width=0.9,
       line_color='white')
show(p)


# There are different glyphs for catogerical data 
#     <ul><li>vbar</li>
#     <li>hbar</li>
#     <li>vbar_stack</li>
#     <li>hbar_stack</li>

# These are some basic concepts and plots that bokeh provides and makes your plot look beautiful and presentable.This plots can be used in notebook as well as browser which makes compatible
