#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# ---
# 

# In this exercise, you'll explore different chart styles, to see which color combinations and fonts you like best!
# 
# ## Setup
# 
# Run the next cell to import and configure the Python libraries that you need to complete the exercise.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# The questions below will give you feedback on your work. Run the following cell to set up our feedback system.

# In[2]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex6 import *
print("Setup Complete")


# You'll work with a chart from the previous tutorial.  Run the next cell to load the data.

# In[3]:


# Path of the file to read
spotify_filepath = "../input/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)


# # Try out seaborn styles
# 
# Run the command below to try out the `"dark"` theme.

# In[8]:


# Change the style of the figure
sns.set_style("ticks")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)

# Mark the exercise complete after the code cell is run
step_1.check()


# Now, try out different themes by amending the first line of code and running the code cell again.  Remember the list of available themes:
# - `"darkgrid"`
# - `"whitegrid"`
# - `"dark"`
# - `"white"`
# - `"ticks"`
# 
# This notebook is your playground -- feel free to experiment as little or as much you wish here!  The exercise is marked as complete after you run every code cell in the notebook at least once.
# 
# ## Keep going
# 
# Learn about how to select and visualize your own datasets in the **[next tutorial](https://www.kaggle.com/alexisbcook/final-project)**!

# ---
# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# 
