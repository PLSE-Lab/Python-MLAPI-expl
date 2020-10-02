#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# ---
# 

# Now it's time for you to demonstrate your new skills with a project of your own!
# 
# In this exercise, you will work with a dataset of your choosing.  Once you've selected a dataset, you'll design and create your own plot to tell interesting stories behind the data!
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


# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# In[ ]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex7 import *
print("Setup Complete")


# ## Step 1: Attach a dataset to the notebook
# 
# Begin by selecting a CSV dataset from [Kaggle Datasets](https://www.kaggle.com/datasets).  If you're unsure how to do this or would like to work with your own data, please revisit the instructions in the previous tutorial.
# 
# Once you have selected a dataset, click on the **[+ ADD DATASET]** option in the top right corner.  This will generate a pop-up window that you can use to search for your chosen dataset.  
# 
# ![ex6_search_dataset](https://i.imgur.com/QDEKwYp.png)
# 
# Once you have found the dataset, click on the **[Add]** button to attach it to the notebook.  You can check that it was successful by looking at the **Workspace** dropdown menu to the right of the notebook -- look for an **input** folder containing a subfolder that matches the name of the dataset.
# 
# ![ex6_dataset_added](https://i.imgur.com/oVlEBPx.png)
# 
# You can click on the carat to the right of the name of the dataset to double-check that it contains a CSV file.  For instance, the image below shows that the example dataset contains two CSV files: (1) **dc-wikia-data.csv**, and (2) **marvel-wikia-data.csv**.
# 
# ![ex6_dataset_dropdown](https://i.imgur.com/4gpFw71.png)
# 
# Once you've uploaded a dataset with a CSV file, run the code cell below **without changes** to receive credit for your work!

# In[ ]:


# Check for a dataset with a CSV file
step_1.check()


# ## Step 2: Specify the filepath
# 
# Now that the dataset is attached to the notebook, you can find its filepath.  To do this, use the **Workspace** menu to list the set of files, and click on the CSV file you'd like to use.  This will open the CSV file in a tab below the notebook.  You can find the filepath towards the top of this new tab.  
# 
# ![ex6_filepath](https://i.imgur.com/pWe0sVb.png)
# 
# After you find the filepath corresponding to your dataset, fill it in as the value for `my_filepath` in the code cell below, and run the code cell to check that you've provided a valid filepath.  For instance, in the case of this example dataset, we would set
# ```
# my_filepath = "../input/dc-wikia-data.csv"
# ```  
# Note that **you must enclose the filepath in quotation marks**; otherwise, the code will return an error.
# 
# Once you've entered the filepath, you can close the tab below the notebook by clicking on the **[X]** at the top of the tab.

# In[ ]:


# Fill in the line below: Specify the path of the CSV file to read
my_filepath = "../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv"
flight_delays_filepath = "../input/data-for-datavis/flight_delays.csv"
# Check for a valid filepath to a CSV file in a dataset
step_2.check()


# ## Step 3: Load the data
# 
# Use the next code cell to load your data file into `my_data`.  Use the filepath that you specified in the previous step.

# In[ ]:


# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath)
flight_delays_data = pd.read_csv(flight_delays_filepath)
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
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("ticks")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=flight_delays_data)

# Check that a figure appears below
step_4.check()


# ## Keep going
# 
# Learn how to use your skills after completing the micro-course to create data visualizations in a **[final tutorial](https://www.kaggle.com/alexisbcook/creating-your-own-notebooks)**.

# ---
# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# 
