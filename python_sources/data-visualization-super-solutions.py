#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization Home Page](https://www.kaggle.com/learn/data-visualization)**
# 
# ---
# 

# 
# 
# ## Scenario
# 
# You've recently decided to create your very own video game!  As an avid reader of [IGN Game Reviews](https://www.ign.com/reviews/games), you hear about all of the most recent game releases, along with the ranking they've received from experts, ranging from 0 (_Disaster_) to 10 (_Masterpiece_).
# 
# ![ex2_ign](https://i.imgur.com/Oh06Fu1.png)
# 
# You're interested in using [IGN reviews](https://www.ign.com/reviews/games) to guide the design of your upcoming game.  Thankfully, someone has summarized the rankings in a really useful CSV file that you can use to guide your analysis.
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


# 

# In[ ]:


# Set up code checking
import os
if not os.path.exists("../input/ign_scores.csv"):
    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex3 import *


# ## Step 1: Load the data
# 
# Read the IGN data file into `ign_data`.  Use the `"Platform"` column to label the rows.

# In[ ]:


# Path of the file to read
ign_filepath = "../input/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data =pd.read_csv(ign_filepath,index_col="Platform")


# ## Step 2: Review the data
# 
# Use a Python command to print the entire dataset.

# In[ ]:


# Print the data
print(ign_data) # Your code here


# The dataset that you've just printed shows the average score, by platform and genre.  Use the data to answer the questions below.

# In[ ]:


# Fill in the line below: What is the highest average score received by PC games,
# for any platform?

high_score = 7.759930

# Fill in the line below: On the Playstation Vita platform, which genre has the 
# lowest average score? Please provide the name of the column, and put your answer 
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
worst_genre = 'Simulation'



# ## Step 3: Which platform is best?
# 
# Since you can remember, your favorite video game has been [**Mario Kart Wii**](https://www.ign.com/games/mario-kart-wii), a racing game released for the Wii platform in 2008.  And, IGN agrees with you that it is a great game -- their rating for this game is a whopping 8.9!  Inspired by the success of this game, you're considering creating your very own racing game for the Wii platform.
# 
# #### Part A
# 
# Create a bar chart that shows the average score for **racing** games, for each platform.  Your chart should have one bar for each platform. 

# In[ ]:


# Bar chart showing average score for racing games by platform
plt.figure(figsize=(10, 6))
# Bar chart showing average score for racing games by platform
sns.barplot(x=ign_data['Racing'], y=ign_data.index)
# Add label for horizontal axis
plt.xlabel("average score")
# Add label for vertical axis
plt.title("Average Score for Racing Games, by Platform")# Your code here

# Check your answer
step_3.a.check()


# #### Part B
# 
# Based on the bar chart, do you expect a racing game for the **Wii** platform to receive a high rating?  If not, what gaming platform seems to be the best alternative?

# ## Step 4: All possible combinations!
# 
# Eventually, you decide against creating a racing game for Wii, but you're still committed to creating your own video game!  Since your gaming interests are pretty broad (_... you generally love most video games_), you decide to use the IGN data to inform your new choice of genre and platform.
# 
# #### Part A
# 
# Use the data to create a heatmap of average score by genre and platform.  

# In[ ]:


#  showing average game score by platform and genre

plt.figure(figsize=(10,10))
# Heatmap showing average game score by platform and genre
sns.heatmap(ign_data, annot=True)
# Add label for horizontal axis
plt.xlabel("Genre")
# Add label for vertical axis
plt.title("Average Game Score, by Platform and Genre")
# Check your answer
step_4.a.check()


# # Keep going
# 
# 
