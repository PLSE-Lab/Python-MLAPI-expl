#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization Home Page](https://www.kaggle.com/learn/data-visualization)**
# 
# ---
# 

# In this exercise, you will use your new knowledge to propose a solution to a real-world scenario. To succeed, you will need to import data into Python, answer questions using the data, and generate **line charts** to understand patterns in the data.
# 
# ## Scenario
# 
# You have recently been hired to manage the museums in the City of Los Angeles. Your first project focuses on the four museums pictured in the images below.
# 
# ![ex1_museums](https://i.imgur.com/pFYL8J1.png)
# 
# You will leverage data from the Los Angeles [Data Portal](https://data.lacity.org/) that tracks monthly visitors to each museum.  
# 
# ![ex1_xlsx](https://i.imgur.com/mGWYlym.png)
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


# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# In[ ]:


# Set up code checking
import os
if not os.path.exists("../input/museum_visitors.csv"):
    os.symlink("../input/data-for-datavis/museum_visitors.csv", "../input/museum_visitors.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex2 import *
print("Setup Complete")


# ## Step 1: Load the data
# 
# Your first assignment is to read the LA Museum Visitors data file into `museum_data`.  Note that:
# - The filepath to the dataset is stored as `museum_filepath`.  Please **do not** change the provided value of the filepath.
# - The name of the column to use as row labels is `"Date"`.  (This can be seen in cell A1 when the file is opened in Excel.)
# 
# To help with this, you may find it useful to revisit some relevant code from the tutorial, which we have pasted below:
# 
# ```python
# # Path of the file to read
# spotify_filepath = "../input/spotify.csv"
# 
# # Read the file into a variable spotify_data
# spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
# ```
# 
# The code you need to write now looks very similar!

# In[ ]:


# Path of the file to read
museum_filepath = "../input/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath , index_col='Date')

# Run the line below with no changes to check that you've loaded the data correctly
step_1.check()


# In[ ]:


# Uncomment the line below to receive a hint
#step_1.hint()
# Uncomment the line below to see the solution
#step_1.solution()


# ## Step 2: Review the data
# 
# Use a Python command to print the last 5 rows of the data.

# In[ ]:


# Print the last five rows of the data 
museum_data.tail()


# The last row (for `2018-11-01`) tracks the number of visitors to each museum in November 2018, the next-to-last row (for `2018-10-01`) tracks the number of visitors to each museum in October 2018, _and so on_.
# 
# Use the last 5 rows of the data to answer the questions below.

# In[ ]:


# Fill in the line below: How many visitors did the Chinese American Museum 
# receive in July 2018?
ca_museum_jul18 = museum_data.loc['2018-07-01','Chinese American Museum'] 

# Fill in the line below: In October 2018, how many more visitors did Avila 
# Adobe receive than the Firehouse Museum?
avila_oct18 = museum_data.loc['2018-10-01','Avila Adobe']-museum_data.loc['2018-10-01','Firehouse Museum']

# Check your answers
step_2.check()


# In[ ]:


# Lines below will give you a hint or solution code
#step_2.hint()
#step_2.solution()


# ## Step 3: Convince the museum board 
# 
# The Firehouse Museum claims they ran an event in 2014 that brought an incredible number of visitors, and that they should get extra budget to run a similar event again.  The other museums think these types of events aren't that important, and budgets should be split purely based on recent visitors on an average day.  
# 
# To show the museum board how the event compared to regular traffic at each museum, create a line chart that shows how the number of visitors to each museum evolved over time.  Your figure should have four lines (one for each museum).
# 
# > **(Optional) Note**: If you have some prior experience with plotting figures in Python, you might be familiar with the `plt.show()` command.  If you decide to use this command, please place it **after** the line of code that checks your answer (in this case, place it after `step_3.check()` below) -- otherwise, the checking code will return an error!

# In[ ]:


# Line chart showing the number of visitors to each museum over time
sns.lineplot(data=museum_data)

# Check your answer
step_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#step_3.hint()
#step_3.solution_plot()


# ## Step 4: Assess seasonality
# 
# When meeting with the employees at Avila Adobe, you hear that one major pain point is that the number of museum visitors varies greatly with the seasons, with low seasons (when the employees are perfectly staffed and happy) and also high seasons (when the employees are understaffed and stressed).  You realize that if you can predict these high and low seasons, you can plan ahead to hire some additional seasonal employees to help out with the extra work.
# 
# #### Part A
# Create a line chart that shows how the number of visitors to Avila Adobe has evolved over time.  (_If your code returns an error, the first thing that you should check is that you've spelled the name of the column correctly!  You must write the name of the column exactly as it appears in the dataset._)

# In[ ]:


# Line plot showing the number of visitors to Avila Adobe over time
sns.lineplot(data=museum_data['Avila Adobe'])

# Check your answer
step_4.a.check()


# In[ ]:


# Lines below will give you a hint or solution code
#step_4.a.hint()
#step_4.a.solution_plot()


# #### Part B
# 
# Does Avila Adobe get more visitors:
# - in September-February (in LA, the fall and winter months), or 
# - in March-August (in LA, the spring and summer)?  
# 
# Using this information, when should the museum staff additional seasonal employees?

# In[ ]:


#step_4.b.hint()


# In[ ]:


# Check your answer (Run this code cell to receive credit!)
step_4.b.solution()

