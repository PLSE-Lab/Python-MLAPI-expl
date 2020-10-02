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
my_filepath ='../input/smart-home-dataset-with-weather-information/HomeC.csv'
# Check for a valid filepath to a CSV file in a dataset
step_2.check()


# ## Step 3: Load the data
# 
# Use the next code cell to load your data file into `my_data`.  Use the filepath that you specified in the previous step.

# ### ADD a Date Time Index to dataset to be more meaningful
#   ####  Dataset use Unix epoch timestamps for min so i calculate the start time to generate the data time index 

# In[ ]:


# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath  ,   parse_dates=True)
home_dat = my_data.select_dtypes(exclude=['object'])

# you can convert a time from unix epoch timestamp to normal stamp using 
# import time 
# print( ' start ' , time.strftime('%Y-%m-%d %H:%S', time.localtime(1451624400)))


time_index = pd.date_range('2016-01-01 05:00', periods=503911,  freq='min')  
time_index = pd.DatetimeIndex(time_index)
home_dat = home_dat.set_index(time_index)
# Check that a dataset has been uploaded into my_data
step_3.check()


# **_After the code cell above is marked correct_**, run the code cell below without changes to view the first five rows of the data.

# # Data Preparation : 

# In[ ]:


energy_data = home_dat.filter(items=[ 'gen [kW]', 'House overall [kW]', 'Dishwasher [kW]',
                                     'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 'Fridge [kW]',
                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',
                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
                                     'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'])

weather_data = home_dat.filter(items=['temperature',
                                      'humidity', 'visibility', 'apparentTemperature', 'pressure',
                                      'windSpeed', 'windBearing', 'dewPoint'])


# In[ ]:


energy_data.head(10)


# In[ ]:


weather_data.head()


# In[ ]:


# Print the first five rows of the data


# ## Step 4: Visualize the data
# 
# Use the next code cell to create a figure that tells a story behind your dataset.  You can use any chart type (_line chart, bar chart, heatmap, etc_) of your choosing!

# ## Generate Data per (day) and (month) :

# In[ ]:


energy_per_day = energy_data.resample('D').sum()
energy_per_day.head()


# In[ ]:


energy_per_month = energy_data.resample('M').sum() # for energy we use sum to calculate overall consumption in period 
plt.figure(figsize=(20,10))
sns.lineplot(data= energy_per_month.filter(items=[ 'Dishwasher [kW]','House overall [kW]',
                                     'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 'Fridge [kW]',
                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',
                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
                                     'Microwave [kW]', 'Living room [kW]', 'Solar [kW]']) , dashes=False  )
# use power == house overall
# gen power == solar 


# ### we can note that :  in August  and September the highest consumption in the year then February month 
# ### and the lowest month in consumption is January

# In[ ]:


plt.figure(figsize=(20,10))
# Plot the rooms consumption 
sns.lineplot(data= energy_per_month.filter(items=[      # remove the devices consumption 
                                     'Home office [kW]',
                                     'Wine cellar [kW]', 'Kitchen 12 [kW]',
                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',
                                      'Living room [kW]']) , dashes=False  )


# ### As we see  the home office has the highest consumption in the home and the living room has the lowest consumption 

# In[ ]:


weather_per_day = weather_data.resample('D').mean()  # note!! (mean) # D =>> for day sample
weather_per_day.head()
weather_per_month = weather_data.resample('M').mean()                # M =>> for month sample


# In[ ]:




plt.figure(figsize=(20,8))

sns.lineplot(data= weather_per_month.filter(items=['temperature',
                                      'humidity', 'visibility', 'apparentTemperature',
                                      'windSpeed', 'dewPoint']) ,dashes=False )




# ### The temprature data not real >> I think there's a problem in temprature sensor 

# In[ ]:


weather_per_month.head()


# ## Home activity in day 2016-10-1  

# In[ ]:


plt.figure(figsize=(20,8))
sns.lineplot(data= energy_data.loc['2016-10-01 00:00' : '2016-10-01 23:00'].filter([ 'Home office [kW]',
                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',
                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
                                 'Living room [kW]']),dashes=False , )


# ### In this day the barn has the highest consumption 

# In[ ]:


plt.figure(figsize=(20,8))
sns.lineplot(data= energy_data.loc['2016-10-01 00:00' : '2016-10-01 23:00'].filter([ 'Home office [kW]',
                                      'Garage door [kW]', 'Kitchen 12 [kW]',
                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]',
                                 'Living room [kW]']),dashes=False , )


# In[ ]:


plt.figure(figsize=(20,8))
sns.lineplot(data= energy_per_day['Solar [kW]'],dashes=False , )


# #### from plot :The Solar power has the highest rate in the April - May 

# In[ ]:


rooms_energy = energy_per_month.filter(items=[      # remove the devices consumption 
                                     'Home office [kW]',
                                     'Wine cellar [kW]', 'Kitchen 12 [kW]',
                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',
                                      'Living room [kW]']) 
devices_energy = energy_per_month.filter(items=[ 'Dishwasher [kW]',
                                     'Furnace 1 [kW]', 'Furnace 2 [kW]',  'Fridge [kW]',
                                     'Garage door [kW]', 
                                     'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
                                     'Microwave [kW]'])

all_rooms_consum = rooms_energy.sum()
all_devices_consum = devices_energy.sum()
print(all_rooms_consum)
print(all_devices_consum)


# In[ ]:


plot = all_rooms_consum .plot(kind = "pie", figsize = (5,5))
plot.set_title("Consumption for room")


# ### from this plot we can see that : the home office has the highest consumption in home

# In[ ]:


plot = all_devices_consum .plot(kind = "pie", figsize = (5,5))
plot.set_title("Consumption for devices")


# ### The furnace has the highest consumption , in devices nearly the half of devices consumption

# In[ ]:


sns.regplot(x=energy_per_day['House overall [kW]'], y= weather_per_day['temperature'])


# ### The temprature effect on home consumption is weak

# In[ ]:


sns.regplot(x=energy_per_day.filter(items = ['Kitchen 12 [kW]','Kitchen 14 [kW]', 'Kitchen 38 [kW]']).sum(axis=1 ), y= weather_per_day['temperature'])


# #### The relation between Kitchens consumption and Temprature are invers

# In[ ]:


sns.regplot(x=energy_per_day['Fridge [kW]'], y= weather_per_day['temperature'])


# ### The relation between temprature and Fridge consumption (Strong dependant)

# In[ ]:


sns.regplot(x=energy_per_day['Garage door [kW]'], y= weather_per_day['temperature'])


# ## Keep going
# 
# Learn how to use your skills after completing the micro-course to create data visualizations in a **[final tutorial](https://www.kaggle.com/alexisbcook/creating-your-own-notebooks)**.

# ---
# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# 

# In[ ]:




