#!/usr/bin/env python
# coding: utf-8

# Now it's time for you to demonstrate your new skills with a project of your own!
# 
# In this exercise, you will work with a dataset of your choosing.  Once you've selected a dataset, you'll design and create your own plot to tell interesting stories behind the data!
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
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex7 import *
print("Setup Complete")


# In[ ]:


# Check for a dataset with a CSV file
step_1.check()


# In[ ]:


# Fill in the line below: Specify the path of the CSV file to read
my_filepath = '../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-40-population-contracted.csv'

# Check for a valid filepath to a CSV file in a dataset
step_2.check()


# ## Step 3: Load the data
# 
# Use the next code cell to load your data file into `my_data`.  Use the filepath that you specified in the previous step.

# In[ ]:


# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath,index_col = "state",parse_dates=True)

# Check that a dataset has been uploaded into my_data
step_3.check()


# **_After the code cell above is marked correct_**, run the code cell below without changes to view the first five rows of the data.

# In[ ]:


# Print the first five rows of the data
my_data.head()


# ## Step 4: Visualize the data
# 
# Use the next code cell to create a figure that tells a story behind your dataset.  You can use any chart type (_line chart, bar chart, heatmap, etc_) of your choosing!

# In[ ]:



# Create a plot
sns.barplot(data=my_data)

# Check that a figure appears below
step_4.check()


# ## Keep going
# 
# Learn how to use your skills after completing the micro-course to create data visualizations in a **[final tutorial](https://www.kaggle.com/alexisbcook/creating-your-own-notebooks)**.

# In[ ]:


plt.figure(figsize=(20,10))
plt.title("Number of Available hospital beds in various states")
sns.barplot(x=my_data.index,y=my_data['available_hospital_beds'])
plt.ylabel("Available Hospital Beds")


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(40,20))

# Add title
plt.title("Number of hospital beds in various states")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(my_data, annot=True)

# Add label for horizontal axis
plt.xlabel("state")


# In[ ]:


# Histogram 
sns.distplot(a=my_data['icu_beds_needed_eighteen_months'], kde=False)


# In[ ]:


sns.kdeplot(data=my_data['icu_bed_occupancy_rate'], shade=True)


# In[ ]:


# 2D KDE plot
sns.jointplot(x=my_data['potentially_available_hospital_beds'], y=my_data['percentage_of_total_icu_beds_needed_six_months'], kind="kde")


# In[ ]:


sns.lineplot(x=my_data['percentage_of_total_icu_beds_needed_six_months'],y='population_65',data=my_data)

