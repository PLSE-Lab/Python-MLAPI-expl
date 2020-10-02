#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# ---
# 

# I used the methods explained in 'Data Visualization: From Non-Coder to Coder Micro Course'
# by Alexis Cook.

# 

# ## Scenario
# Using [IGN reviews](https://www.ign.com/reviews/games) to guide the design of upcoming game. 
# The data is in CSV file that we can use to guide our analysis.
# 
# ## Setup
# 
# Run the next cell to import and configure the Python libraries that you need to complete the exercise.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# ## Step 1: Load the data
# 
# Read the IGN data file into `ign_data`.  Use the `"Platform"` column to label the rows.

# In[ ]:


# Path of the file to read
ign_filepath = "../input/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")


# ## Step 2: Review the data
# 
# Use a Python command to print the entire dataset.

# In[ ]:


# Print the data
ign_data # Your code here


# The dataset that you've just printed shows the average score, by platform and genre.  Use the data to answer the questions below.

# Data types of each column

# In[ ]:


ign_data.info()


# In[ ]:


#Easiet way to explore each column of the dataser is using describe command
ign_data.describe()


# ## Step 3: Which platform is best?
# 
# #### Part A
# 
# Create a bar chart that shows the average score for **racing** games, for each platform

# In[ ]:


# Bar chart showing average score for racing games by platform
# Set the width and height of the figure
plt.figure(figsize=(8, 6))
# Bar chart showing average score for racing games by platform
sns.barplot(x=ign_data['Racing'], y=ign_data.index)
# Add label for horizontal axis
plt.xlabel("")
# Add label for vertical axis
plt.title("Average Score for Racing Games, by Platform")


# Based on the data, we should not expect a racing game for the Wii platform to receive a high rating. In fact, on average, racing games for Wii score lower than any other platform. Xbox One seems to be the best alternative, since it has the highest average ratings.

# ## Step 4: Supporting Decision_Making Process with Data analysis
# 
# use the IGN data to decide which genre and platform is the best choice.
# 
# #### Part A
# 
# Use the data to create a heatmap of average score by genre and platform.  

# In[ ]:


# Heatmap showing average game score by platform and genre
# Set the width and height of the figure
plt.figure(figsize=(10,10))
# Heatmap showing average game score by platform and genre
sns.heatmap(ign_data, annot=True)
# Add label for horizontal axis
plt.xlabel("Genre")
# Add label for vertical axis
plt.title("Average Game Score, by Platform and Genre")


# Simulation games for Playstation 4 receive the highest average ratings (9.2). Shooting and Fighting games for Game Boy Color receive the lowest average rankings (4.5).

# In[ ]:





# # Keep going
# 
# Move on to learn all about **[scatter plots](https://www.kaggle.com/alexisbcook/scatter-plots)**!

# ---
# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# 
