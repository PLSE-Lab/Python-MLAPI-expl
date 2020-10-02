#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization Home Page](https://www.kaggle.com/learn/data-visualization)**
# 
# ---
# 

# In this micro-course, you've learned how to create many different chart types.  Now, you'll organize your knowledge, before learning some quick commands that you can use to change the style of your charts.

# # What have you learned?
# 
# <img src="https://imgur.com/2VmgDnF.png" height="500" width="1000" usemap="#plottingmap" />
# <map name="plottingmap">
#   <area shape="rect" coords="262,342,402,476" href="https://www.kaggle.com/alexisbcook/hello-seaborn" title="EXAMPLE: sns.lineplot(data=my_data)">
#   <area shape="rect" coords="8,75,154,200" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title="EXAMPLE: sns.swarmplot(x=my_data['Column 1'], y=my_data['Column 2'])">
#    <area shape="rect" coords="8,200,154,350" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title="EXAMPLE: sns.regplot(x=my_data['Column 1'], y=my_data['Column 2'])">
#    <area shape="rect" coords="8,350,154,500" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title='EXAMPLE: sns.lmplot(x="Column 1", y="Column 2", hue="Column 3", data=my_data)'>
#       <area shape="rect" coords="229,10,393,160" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title="EXAMPLE: sns.scatterplot(x=my_data['Column 1'], y=my_data['Column 2'], hue=my_data['Column 3'])">
#      <area shape="rect" coords="397,10,566,160" href="https://www.kaggle.com/alexisbcook/line-charts" title="EXAMPLE: sns.heatmap(data=my_data)">
#      <area shape="rect" coords="565,10,711,160" href="https://www.kaggle.com/alexisbcook/line-charts" title="EXAMPLE: sns.barplot(x=my_data.index, y=my_data['Column'])">
#      <area shape="rect" coords="780,55,940,210" href="https://www.kaggle.com/alexisbcook/scatter-plots" title="EXAMPLE: sns.jointplot(x=my_data['Column 1'], y=my_data['Column 2'], kind='kde')">
#      <area shape="rect" coords="780,210,940,350" href="https://www.kaggle.com/alexisbcook/scatter-plots" title="EXAMPLE: sns.kdeplot(data=my_data['Column'], shade=True)">
#    <area shape="rect" coords="780,360,1000,500" href="https://www.kaggle.com/alexisbcook/scatter-plots" title="EXAMPLE: sns.distplot(a=my_data['Column'], kde=False)">
# </map>
# 
# Since it's not always easy to decide how to best tell the story behind your data, we've broken the chart types into three broad categories to help with this.
# - **Trends** - A trend is defined as a pattern of change.
#     - `sns.lineplot` - **Line charts** are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.
# - **Relationship** - There are many different chart types that you can use to understand relationships between variables in your data.
#     - `sns.barplot` - **Bar charts** are useful for comparing quantities corresponding to different groups.
#     - `sns.heatmap` - **Heatmaps** can be used to find color-coded patterns in tables of numbers.
#     - `sns.scatterplot` - **Scatter plots** show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third [categorical variable](https://en.wikipedia.org/wiki/Categorical_variable).
#     - `sns.regplot` - Including a **regression line** in the scatter plot makes it easier to see any linear relationship between two variables.
#     - `sns.lmplot` - This command is useful for drawing multiple regression lines, if the scatter plot contains multiple, color-coded groups.
#     - `sns.swarmplot` - **Categorical scatter plots** show the relationship between a continuous variable and a categorical variable.
# - **Distribution** - We visualize distributions to show the possible values that we can expect to see in a variable, along with how likely they are.
#     - `sns.distplot` - **Histograms** show the distribution of a single numerical variable.
#     - `sns.kdeplot` - **KDE plots** (or **2D KDE plots**) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
#     - `sns.jointplot` - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.
#     
# # Changing styles with seaborn
# 
# All of the commands have provided a nice, default style to each of the plots.  However, you may find it useful to customize how your plots look, and thankfully, this can be accomplished by just adding one more line of code!
# 
# As always, we need to begin by setting up the coding environment.  (_This code is hidden, but you can un-hide it by clicking on the "Code" button immediately below this text, on the right._)

# In[ ]:



import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# We'll work with the same code that we used to create a line chart in a previous tutorial.  The code below loads the dataset and creates the chart.

# In[ ]:


# Path of the file to read
spotify_filepath = "../input/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# We can quickly change the style of the figure to a different theme with only a single line of code.  

# In[ ]:


# Change the style of the figure to the "dark" theme
sns.set_style("dark")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# Seaborn has five different themes: (1)`"darkgrid"`, (2)`"whitegrid"`, (3)`"dark"`, (4)`"white"`, and (5)`"ticks"`, and you need only use a command similar to the one in the code cell above (with the chosen theme filled in) to change it.  
# 
# In the upcoming exercise, you'll experiment with these themes to see which one you like most!
# 
# # What's next?
# 
# Explore seaborn styles in a quick **[coding exercise](https://www.kaggle.com/kernels/fork/2959763)**!

# ---
# **[Data Visualization Home Page](https://www.kaggle.com/learn/data-visualization)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161291) to chat with other Learners.*
