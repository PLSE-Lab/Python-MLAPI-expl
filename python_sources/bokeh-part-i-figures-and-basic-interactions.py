#!/usr/bin/env python
# coding: utf-8

# # Bokeh Part I - Figures and basic Interactions

# **Welcome** and thanks for your interest in my python **bokeh beginners tutorial**
# 
# 
# This kernel is actually my approach to get familiar with bokeh and widen my visualization skills. If this kernel is also a help to you, than this is a big bonus.
# 
# 
# This work was inspired by:
# * https://realpython.com/python-data-visualization-bokeh/ 
# * https://www.kaggle.com/pavlofesenko/interactive-titanic-dashboard-using-bokeh from [@pavlofesenko](https://www.kaggle.com/pavlofesenko) 
# * https://www.kaggle.com/tavoosi/suicide-data-full-interactive-dashboard from [@tavoosi](https://www.kaggle.com/tavoosi)

# # Introduction
# 
# For the visualizations in this notebook I rely heavily on [bokeh](https://bokeh.org/) which in my oppinion is one of the most intuitive python library for easy creation of interactive visualizations. For a more detailed and in depth description you can visit Bokeh's great [user guide](https://bokeh.pydata.org/en/latest/docs/user_guide.html) and excellent [gallery](https://docs.bokeh.org/en/latest/docs/gallery.html).

# The first step is to load all libraries we are going to use in the sections following. Without data, there won't be anything to visualize and interact, so we will import the usual suspects *Numpy* and *Pandas*.
# 
# Every visualization more or less contains the same steps, which are also reflected in the bokeh modules shown here. 
# 1. First of all, the figure needs to be set up: 
#     * from bokeh.plotting import figure
# 2. The data needs to be connected and drawn to the figure: 
#     * from bokeh.models import ColumnDataSource, HoverTool
# 3. Orchestrate the way you want to layout your figure(s). This is important if you create more than one figure
#     * from bokeh.layouts import row, column, gridplot
# 4. Finally import the modules needed to render your figures either to a file or like here to our notebook
#     * from bokeh.io import output_file, output_notebook
#     
# With the libraries in place, we can start to have fun. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Data Handling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Bokeh Libraries
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_file, output_notebook

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# One important step is to tell bokeh where to render the visualization. We want to interact with them through our notebook.

# In[ ]:


output_notebook()


# Last but not least we need the data we want to interact with. For the purpose of this tutorial we will import the [UCI heart disease data](https://www.kaggle.com/ronitf/heart-disease-uci).

# In[ ]:


# prepare the data
path_to_data = '/kaggle/input/heart-disease-uci/heart.csv'
heart_data = pd.read_csv(path_to_data)
print(type(heart_data))
heart_data.head()


# Now we have all the ingredients needed. So let's start and prepare our first simple visualization.

# # 1. A Simple Figure

# Bokeh can digest python dict, pandas DataFrames as well as pandas groupby so there is no problem in using them right away for our visualization. Never the less, Bokeh comes with its own data format called **ColumnDataSource**. To unleash Bokeh's full interaction superpowers the input data needs to be of type ColumnDataSource. So in my opinion it is a good habit to work with ColumnDataSource right away, even for simple visualizations.

# In[ ]:


heart_cds = ColumnDataSource(heart_data)


# By setting up the figure, we determine the overall look of the visualization. This is like the environment, which we later populate with our data. Here we give the overall information like the title, dimensions and captions. One important parameter is toolbar_location, which tells the figure where to place the toolbar. Or not like in our case. The toolbar is the simplest way to add interactivity to our visualization. We will add a toolbar in a later step.

# In[ ]:


# set-up the figure
fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",
            plot_height=400, plot_width=600,
            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',
            toolbar_location=None)


# Next we have to fill our figure with life, so let's add some data. As you might guess by the axis-labels we want to visualize the relationship of cholesterol dependent of age.
# The available glyphs to plot your data are many and I recommend everybody to have a deeper look at the [reference for plotting](https://docs.bokeh.org/en/latest/docs/reference/plotting.html). For now, we just want to create a simple visualization and stay with circles.

# In[ ]:


# Connect to and draw the data
fig.circle(x='age', y='chol', source=heart_cds)


# So far we set-up our figure and connected our data to draw it. Finaly we have to tell bokeh that we were done with our visualization and want it to be shown. 

# In[ ]:


# Preview and Save the Figure
show(fig)


# Oooohh, that was easy. In just a few lines of code we created a decent looking visualisation. And maybe some of you already noticed, it already contains a little interaction, just click on the figure and move around.
# 
# **Now I want more!** I like to see more information, especially if we had a true heart disease. This is as easy as the previous example ;)
# 
# Let's start with a new prepare data, set-up figure, connect to and draw the data circle.
# This time, we distinguish between actual appearance of a heart disease and a false positive.
# 
# Just to show, that pandas DataFrames work to, I neglect to create ColumnDataSources for now.

# In[ ]:


# Prepare the data
# Create the cds for each outcome
disease_data = heart_data[heart_data['target']==1]
healthy_data = heart_data[heart_data['target']==0]


# In[ ]:


# set-up the figure
fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",
            plot_height=400, plot_width=600,
            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',
            toolbar_location=None)


# For the sake of this tutorial, I will plot two different glyphs with different colors. 
# 
# Note:In the second part of my Bokeh tutorial (On styling and more complex interactions) I will use transformations and mappings.

# In[ ]:


# Connect to and draw the data
fig.circle(x='age', y='chol', source=disease_data, color='red', size=7, legend_label='Heart Disease')
fig.circle(x='age', y='chol', source=healthy_data, color='green', size=7, legend_label='False Alarm')


# In[ ]:


# Preview and Save the Figure
show(fig)


# I actually want to put in one **more Information**. Is there a difference between males and females? Let's try other glyphes, too. 
# 
# But actually, everything stays the same:
# 1. Prepare the data
# 2. Set up the figure
# 3. Connect to and draw data
# 4. Organize layout (Nothing to do, yet ;)
# 5. Show visualization

# In[ ]:


# Prepare the data
# Create the cds for each outcome
female_disease_data = disease_data[disease_data['sex']==0]
male_disease_data = disease_data[disease_data['sex']==1]
female_healthy_data = healthy_data[healthy_data['sex']==0]
male_healthy_data = healthy_data[healthy_data['sex']==1]


# In[ ]:


# set-up the figure
fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",
            plot_height=400, plot_width=600,
            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',
            toolbar_location=None)


# To draw our data, I choose to use circles for females and diamonds for males. But when using the same size on the different glyphs, the circles appear more dominant. To capture this, we set different sizes for the glyphs.

# In[ ]:


# Connect to and draw the data
fig.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease')
fig.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease')
fig.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm')
fig.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm')


# In[ ]:


# Preview and Save the Figure
show(fig)


# Ok, this still was easy. Nothing new, but I am a little unhappy with the result. Now it get's a little messy and hard to read. So let us add some interactivity to make this plot more useful.

# # 2. Adding simple Interactions

# As we just saw, things can get hard to see when we add information to our visualizations. Interactions are a nice way to filter the information on what we are interested.
# 
# Adding simple interactions in Bokeh is easy and needs just a little alteration to our previous code.
# 
# Again, we start over with our prepare data, set up figure, connect and draw data circle. 

# In[ ]:


# Prepare the data
# Create the cds for each outcome
female_disease_data = disease_data[disease_data['sex']==0]
male_disease_data = disease_data[disease_data['sex']==1]
female_healthy_data = healthy_data[healthy_data['sex']==0]
male_healthy_data = healthy_data[healthy_data['sex']==1]

# set-up the figure
fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",
            plot_height=400, plot_width=600,
            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',
            toolbar_location=None)


# For the beginning, let us mute data points by selecting the appropriate group on the legend. So we will add the parameter muted_alpha to our glyphs and assign an alpha value to make them almost invisible. Then we add a click policy to the legend by simply one line of code: fig.legend.click_policy = 'mute'. If you want the data points to disappear just change the policy from 'mute' to 'hide'

# In[ ]:


# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend
# Connect to and draw the data
fig.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)
fig.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)
fig.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)
fig.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)

# add a policy to the legend what should happen when it is clicked
fig.legend.click_policy = 'mute' #'hide'


# In[ ]:


# Preview and Save the Figure
show(fig)


# Like in the steps before, it is that easy. But I want even more information, this time on a micro level. So how can I extract the information of one specific point in my plot?
# As before, bokeh makes it pretty simple with their module HoverTool. So let's set up our figure.

# In[ ]:


fig_hover = figure(title="Bokeh Visualisation of UCI Heart Disease Data",
                    plot_height=400, plot_width=600,
                    x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',
                    toolbar_location=None)
# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend
# Connect to and draw the data
fig_hover.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)
fig_hover.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)
fig_hover.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)
fig_hover.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)

# add a policy to the legend what should happen when it is clicked
fig_hover.legend.click_policy = 'mute' #'hide'


# Now we define what information to display on hover over and assign it to our figure.

# In[ ]:


# define information to display
tool_tips = [
             ('Age', '@age'),
             ('Sex', '@sex'),
             ('Pain Type', '@cp'),
             ('Max Heartrate', '@thalach')
            ]

# connect to the figure
fig_hover.add_tools(HoverTool(tooltips=tool_tips))


# In[ ]:


# Preview and Save the Figure
show(fig_hover)


# That's enough information for one picture. So let's add another one to get more information. In the second figure we will look at the blood presure compared to the age.
# 
# We can keep most of the code we wrote before. The data stays the same as before and our mute on clicking the legend functionality.

# In[ ]:


# Creating the plots we actualy already have
# set-up the figure
cholesterol_fig = figure(title="Correlation between age and cholesterol",
            plot_height=400, plot_width=600,
            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',
            toolbar_location=None)
# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend
# Connect to and draw the data
cholesterol_fig.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)
cholesterol_fig.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)
cholesterol_fig.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)
cholesterol_fig.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)

cholesterol_fig.legend.click_policy = 'mute'


# In[ ]:


# Create the new plot
# set-up the figure
bloodpres_fig = figure(title="Correlation between age and blood preasure",
            plot_height=400, plot_width=600,
            x_axis_label='age [y]', y_axis_label='blood preasure [mm Hg]',
            toolbar_location=None)
# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend
# Connect to and draw the data
bloodpres_fig.circle(x='age', y='trestbps', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)
bloodpres_fig.diamond(x='age', y='trestbps', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)
bloodpres_fig.circle(x='age', y='trestbps', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)
bloodpres_fig.diamond(x='age', y='trestbps', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)

bloodpres_fig.legend.click_policy = 'mute'


# Now we actually have two figures we want to visualize. Here the Bokeh layouts row, column and gridplot come into play. For an in depth study of the posibility to configure your visualizations layouts I recommend the [bokeh.layouts](https://docs.bokeh.org/en/latest/docs/reference/layouts.html) reference.
# 
# We want to plot our figures next to each other, so we use the row method.

# In[ ]:


# Organize the layout
row_layout = row([cholesterol_fig, bloodpres_fig])
# Preview and Save the Figure
show(row_layout)


# This again was easy, but I am not happy with it.
# * It is hard to compare the points given the age.
# * If I move around in one figure, the other stays as it is.
# * I cannot select data points for a more detailed inspection
# 
# So let us address this points by switching from row-layout to column-layout and add some interaction between those two figures. 
# 
# So let's check out, how a selection of points in one figure is carried over to the other figure. Like muting points, this is also really easy to do with Bokeh. The key point is the above mentioned ColumnDataSource. In creating a ColumnDataSource out of our pandas DataFrame, we can share information between different figures. We also add tools which provide a simple way to get interactions like selection, pan or zoom.

# In[ ]:


# Prepare the data
# Create the cds for each outcome
female_disease_cds = ColumnDataSource(disease_data[disease_data['sex']==0])
male_disease_cds = ColumnDataSource(disease_data[disease_data['sex']==1])
female_healthy_cds = ColumnDataSource(healthy_data[healthy_data['sex']==0])
male_healthy_cds = ColumnDataSource(healthy_data[healthy_data['sex']==1])


# Now we add tools to our visualization and assign them to our figures.

# In[ ]:


# Specify the tools
tool_list = ['pan', 'wheel_zoom', 'box_select', 'reset']

# Creating the plots we actualy already have
# set-up the figure
cholesterol_fig = figure(title="Correlation between age and cholesterol",
            plot_height=300, plot_width=900,
            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',
            tools=tool_list)

# set-up the figure
bloodpres_fig = figure(title="Correlation between age and blood preasure",
            plot_height=300, plot_width=900,
            x_axis_label='age [y]', y_axis_label='blood preasure [mm Hg]',
            tools=tool_list)


# Again connect and draw the data.

# In[ ]:



# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend
# Connect to and draw the data
cholesterol_fig.circle(x='age', y='chol', source=female_disease_cds, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)
cholesterol_fig.diamond(x='age', y='chol', source=male_disease_cds, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)
cholesterol_fig.circle(x='age', y='chol', source=female_healthy_cds, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)
cholesterol_fig.diamond(x='age', y='chol', source=male_healthy_cds, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)

cholesterol_fig.legend.click_policy = 'mute'

# Create the new plot

# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend
# Connect to and draw the data
bloodpres_fig.circle(x='age', y='trestbps', source=female_disease_cds, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)
bloodpres_fig.diamond(x='age', y='trestbps', source=male_disease_cds, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)
bloodpres_fig.circle(x='age', y='trestbps', source=female_healthy_cds, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)
bloodpres_fig.diamond(x='age', y='trestbps', source=male_healthy_cds, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)

bloodpres_fig.legend.click_policy = 'mute'


# Now we link together the x-axes to mirror changes on the selected x-axis of one figure to the other.

# In[ ]:


#now we link together the x-axes
cholesterol_fig.x_range = bloodpres_fig.x_range


# Finally we aligne the figures for easy comparison compared to the age.

# In[ ]:


# Aligne the figures column wise
column_layout = column([cholesterol_fig, bloodpres_fig])
# Preview and Save the Figure
show( column_layout )


# # Conclusion Part I
# I realy enjoyed writing this tutorial and learned a lot. It surprised me that it is this easy to start with and get some interaction with the visualisations. But to be honest, I am not satisfied, yet. For high quality visualizations there is needed more. So for my second Part I will tackle more useful and complex visualizations both in functionality and style.
# 
# If you too learned something, I am happy. If you liked it or want to give some constructive feedback you're welcome.
# 
# Cheers,
#    Borschi
# 

# In[ ]:




