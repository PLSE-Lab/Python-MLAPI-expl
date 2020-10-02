#!/usr/bin/env python
# coding: utf-8

# In this exercise, you'll explore different chart styles, to see which color combinations and fonts you like best!
# 
# ## Setup
# 
# Run the next cell to import and configure the Python libraries that you need to complete the exercise.

# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# The questions below will give you feedback on your work. Run the following cell to set up our feedback system.

# In[ ]:


# Set up code checking
import os
if not os.path.exists("../input/spotify.csv"):
    os.symlink("../input/data-for-datavis/spotify.csv", "../input/spotify.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex6 import *
print("Setup Complete")


# You'll work with a chart from the previous tutorial.  Run the next cell to load the data.

# In[ ]:


# Path of the file to read
spotify_filepath = "../input/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)


# # Try out seaborn styles
# 
# Run the command below to try out the `"dark"` theme.

# In[ ]:


# Change the style of the figure
sns.set_style("white")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)

# Mark the exercise complete after the code cell is run
step_1.check()

