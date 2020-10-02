#!/usr/bin/env python
# coding: utf-8

# # MEI Introduction to Data Science
# # Lesson 3 - Activity 2
# The problem to be solved from the dataset used in this activity requires the data to be cleaned by removing unwanted rows, replace values and filter a dataset. The activity uses the data from the AQA large data set which gives information about cars.

# ## Problem
# > *Are petrol or diesel cars heavier and which have higher emissions?*
# 
# To answer this question you could find the mean and standard deviation of the masses and emissions of petrol and diesel cars in the dataset.

# ## Getting the data
# This activity uses the data from the AQA large data set which gives information about cars.
# 
# * Run the code to import the data

# In[ ]:


# import pandas
import pandas as pd

# importing the data
cars_data=pd.read_csv('../input/aqalds/AQA-large-data-set.csv')

# inspecting the dataset to check that it has imported correctly
cars_data.head()


# ## Exploring the data
# ### Checking the data and removing unwanted rows
# With any dataset you should perform some checks to ensure that the data imported for a field have suitable values. The code below uses the `describe` command to get a summary of the data in the Mass field.
# 
# * Run the code below
# * Explain why a value of** **0 might have been recorded for the mass of a car

# In[ ]:


print(cars_data['Mass'].describe())


# The value of 0 is not a suitable value for the mass of a car. The code below completely removes any of these rows by rewriting a copy of the data set with only those value where Mass>0.
# 
# * Run the code below

# In[ ]:


# A new copy of the cars_data dataset created 
# This contains only those values from the original cars_data dataset where the value of the Mass is >0
cars_data=cars_data[cars_data['Mass'] >0]

# describe is used to check that these values look suitable
print(cars_data['Mass'].describe())


# **Checkpoint**
# > * Why might values of 0 have been recorded for the masses of some of the cars?
# > * Are the statistics given by `describe` suitable for the masses of cars? Give some supporting evidence for your answer.

# ### Replacing values so the data can be explored in different categories
# The PropulsionTypeId field contains a code which references the engine type. 
# 
# The code below replaces these values with the actual descriptions.	<br>
# 1:	Petrol<br>
# 2:	Diesel<br>
# 3:	Electric<br>
# 7:	Gas/Petrol<br>
# 8:	Electric/Petrol
# * Run the code below to update the propulsion type

# In[ ]:


# the PropulsionTypeId field is overwritten with the values from the list using the replace command
# the replace list uses a colon to indicate what is to be replaced and a commas to separate the items
cars_data['PropulsionTypeId'] = cars_data['PropulsionTypeId'].replace({1: 'Petrol',
                                                                       2: 'Diesel',
                                                                       3: 'Electric', 
                                                                       7: 'Gas/Petrol', 
                                                                       8: 'Electric/Petrol'})
cars_data.head()


# You might also find it useful to see the boxplots for these fields. The code below will generate the boxplots grouped by propulsion type. Plotting charts will be explored in more detail in lesson 4.

# In[ ]:


# import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


cars_data.boxplot(column = ['Mass'],by='PropulsionTypeId', vert=False,figsize=(12, 8))
plt.show()

cars_data.boxplot(column = ['CO2'],by='PropulsionTypeId', vert=False,figsize=(12, 8))
plt.show()


# ## Analysing the data
# ### Filtering data 
# To filter the data you can create a new dataset in the Notebook that contains only the rows where a required condition holds. In the text below the line:<br>
# `petrol_data = cars_data[cars_data['PropulsionTypeId'] == 'Petrol']`<br>
# creates a new data set called `petrol_data` that contains only the rows for which the propulsion type is "Petrol".
# 

# In[ ]:


# NOTE: This block of code will only work if the PropulsionTypeId has been changed to the text values

# a new datset called petrol_data is created that contains only the rows where the Propulsion Type is Petrol
petrol_data = cars_data[cars_data['PropulsionTypeId'] == 'Petrol']
petrol_data.head()


# You can use the `print()` command to print out multiple rows of text. You can create a line of text by adding together separate strings using `+`. 
# 
# All the items added must be strings so you can't just use `print("mass: mean = "+petrol_data['Mass'].mean())` as `petrol_data['Mass'].mean()` is a floating point number. You can convert this to a string using `str(petrol_data['Mass'].mean())` and then use this to build the line of text.
# * Run the code
# * Confirm that the values seem appropriate for the masses of petrol cars
# * Add code to also calculate the mean and standard deviation of the CO2 emissions

# In[ ]:


# print out the mean and standard deviation for the Mass converted to text (i.e. a string)
print("mass: mean = "+str(petrol_data['Mass'].mean()))
print("mass: standard deviation = "+str(petrol_data['Mass'].std()))


# * Update and run the code below so that it creates a new data set of the diesel cars and calculates the mean and standard deviation of their masses and CO2 emissions.

# In[ ]:


# NOTE: This block of code will only work if the PropulsionTypeId has been changed to the text values

# a new datset called petrol_data is created
# this contains only the rows where the Propulsion Type is Petrol
petrol_data = cars_data[cars_data['PropulsionTypeId'] == 'Petrol']

# print out the mean and standard deviation for the Mass converted to text (i.e. a string)
print("mass: mean = "+str(petrol_data['Mass'].mean()))
print("mass: standard deviation = "+str(petrol_data['Mass'].std()))


# ## Communicating the results
# **Checkpoint**
# > Use the measures calculated above to answer the original problem: *Are petrol or diesel cars heavier and which have higher emissions?*
