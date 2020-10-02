#!/usr/bin/env python
# coding: utf-8

# **Bokeh 101**

# *This notebook represents the first step of my learning journey about Bokeh. I included my learning journey resources in the references section.*
# 
# **What is Bokeh?**
# 
# Bokeh is a library for creating interactive data visualizations. It offers a concise, human-readable syntax, which allows for rapidly presenting data in an aesthetically pleasing manner. 
# 
# 

# In[ ]:


#importing Bokeh
from bokeh.plotting import figure
from bokeh.io import output_file, show, output_notebook
output_notebook()


# *** Line Glyphs**
# 
# Single Lines

# In[ ]:


#prepare some data
x = [1,2,3,4,5]
y = [6,7,8,9,10]
#create a figure object
f = figure(plot_width=400, plot_height=400)
#create line plot
f.line(x,y,line_width=2)
#write the plot in the figure object
show(f)


# Step Line

# In[ ]:


f = figure(plot_width=400, plot_height=400)

# add a steps renderer
f.step([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2, mode="center")

show(f)


# Multiple Lines

# In[ ]:


f = figure(plot_width=400, plot_height=400)

f.multi_line([[1, 3, 2], [3, 4, 6, 6]], [[2, 1, 4], [4, 7, 8, 5]],
             color=["purple", "green"], alpha=[0.8, 0.3], line_width=4)

show(f)


# **Bars**
# 
# Bokeh provides the hbar() and vbar() glyphs function for this purpose.

# In[ ]:


f = figure(plot_width=400, plot_height=400)
f.vbar(x=[1, 2, 3], width=0.5, bottom=0,
       top=[1.2, 2.5, 3.7], color="purple")

show(f)


# In[ ]:


f = figure(plot_width=400, plot_height=400)
f.hbar(y=[1, 2, 3], height=0.5, left=0,
       right=[1.2, 2.5, 3.7], color="green")

show(f)


# **Bokeh Sample Data**
# 
# Bokeh allows us to practice with sample data sets. iris is one of them.

# In[ ]:


from bokeh.sampledata.iris import flowers


# In[ ]:


# Print the first 5 rows of the data
flowers.head()


# In[ ]:


# Print the last 5 rows of the data
flowers.tail()


# In[ ]:


colormap={'setosa':'red','versicolor':'green','virginica':'blue'}
flowers['color'] = [colormap[x] for x in flowers['species']]
flowers['size'] = flowers['sepal_width'] * 4


# In[ ]:


#after adding color and size columns
flowers.head()


# **ColumnDataSource**
# 
# The ColumnDataSource is the core of most Bokeh plots, providing the data that is visualized by the glyphs of the plot. With the ColumnDataSource, it is easy to share data between multiple plots and widgets, such as the DataTable.A ColumnDataSource is simply a mapping between column names and lists of data. 

# In[ ]:


from bokeh.models import ColumnDataSource

setosa = ColumnDataSource(flowers[flowers["species"]=="setosa"])
versicolor = ColumnDataSource(flowers[flowers["species"]=="versicolor"])
virginica = ColumnDataSource(flowers[flowers["species"]=="virginica"])


# In[ ]:


#Create the figure object
f = figure(plot_width=1000, plot_height=400)

#adding glyphs
f.circle(x="petal_length", y="petal_width", size='size', fill_alpha=0.2, 
color="color", line_dash=[5,3], legend_label='Setosa', source=setosa)

f.circle(x="petal_length", y="petal_width", size='size', fill_alpha=0.2, 
color="color", line_dash=[5,3], legend_label='Versicolor', source=versicolor)

f.circle(x="petal_length", y="petal_width", size='size', fill_alpha=0.2,
color="color", line_dash=[5,3], legend_label='Virginica', source=virginica)

show(f)


# In[ ]:


#Style the legend
f.legend.location = (500,500)
f.legend.location = 'top_left'
f.legend.background_fill_alpha = 0
f.legend.border_line_color = None
f.legend.margin = 10
f.legend.padding = 18
f.legend.label_text_color = 'black'
f.legend.label_text_font = 'times'

show(f)


# **References**
# 
# 1. https://docs.bokeh.org
# 2. Data Visualization on the Browser with Python and Bokeh - Udemy Course -  [udemy.com/course/python-bokeh/]
# 3. Kaggle - Micro Courses - Data Visualization

# In[ ]:




