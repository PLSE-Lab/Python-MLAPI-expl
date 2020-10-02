#!/usr/bin/env python
# coding: utf-8

# Hello Kagglers,
# 
# This notebook is actually forked from a data visualization course by [Alexis Cook](https://www.kaggle.com/alexisbcook). If you are new to Machine Learning and you want to understand the basics of how you can do EDA on the data and view the results in the form of a Chart then you must complete this [course](https://www.kaggle.com/learn/data-visualization)
# 
# Now, in this notebook you will find various charts and startup code for the same.
# 
# As the instructor have divided the charts in following 3 categories, it will help you to understand it nicely.
# 
# 3 categories which can help you to decide which chart can give you insights for a given scenario:-
# 1. Trends
#     a. Line charts     
# 2. Relationshio
#     a. Bar charts
#     b. Scatter plots
#     c. Regression plots
#     d. Swarm plots
#     e. LM plots
#     f. Heat maps
# 3. Distrubution
#     a. Histogram
#     b. Kenrnel Density Plot
#     
# Refer this notebook and I highly recommend to do the [course](https://www.kaggle.com/learn/data-visualization). Also, please upvote if you like the notebook.

# ## Data Visualization:
# 
# ## Type 1: Seaborn example - Fifa Match Dataset
# 
# ### Step 1. Load the necessary libraries and file path

# In[ ]:



import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Set up code checking
import os
if not os.path.exists("../input/fifa.csv"):
    os.symlink("../input/data-for-datavis/fifa.csv", "../input/fifa.csv")  
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex1 import *
print("Setup Complete")


# ### Step 2: Load the data
# 
# You are ready to get started with some data visualization! 

# In[ ]:


# Path of the file to read
fifa_filepath = "../input/fifa.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)
print (fifa_data.info())


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(16,10))
plt.xlabel("Years", fontsize=18)
plt.ylabel("Rank in the Tournament")

#Either do below steps or
#plt.plot(fifa_data)
#plt.show()

#Or use lineplot
# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)


# 
# ## Type 2: Line Charts - Spotify Dataset
# 
# ### Step 1. Load the necessary libraries and file path

# In[ ]:


# Path of the file to read
spotify_filepath = "../input/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

spotify_data.head(10)


# In[ ]:


spotify_data.tail(10)


# In[ ]:


#Basic view
sns.lineplot(data = spotify_data)


# In[ ]:


#Now let's see the map in an enlarged view
plt.figure(figsize=(14,6))
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
plt.xlabel("Dates")
plt.ylabel("No. of streams")
sns.lineplot(data = spotify_data)


# In[ ]:


#How to view subset of a dataset on line chart
plt.title("Subset view of the entire dataset")
sns.lineplot(data = spotify_data["Shape of You"], label = "Shape of You")
sns.lineplot(data = spotify_data["Despacito"], label="Despacito")


# ## 3. Bar Charts and Heatmaps - US Flights Delay dataset
# 
# ### Step 1: Load necessary libararies and dataset
# 
# 

# In[ ]:


flight_filepath = "../input/flight_delays.csv"

flight_data = pd.read_csv(flight_filepath, index_col = "Month")


# In[ ]:


#Viewind the data of the file
flight_data


# ### Step 2. View the data in Barchart

# In[ ]:


plt.figure(figsize = (15, 6))
plt.title("Monthwise average arrival delay of the flight")

sns.barplot(x = flight_data.index, y = flight_data["NK"])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")


# ### 3. View data in Heatmap

# In[ ]:


plt.figure(figsize = (14, 7))
plt.title("Average arrival delay fo each airline, monthwise")

sns.heatmap(data = flight_data, annot = True)

plt.xlabel("Airlines")


# ## 4. Scatter Plots - Insurance dataset
# 
# ### Step 1 - setup and add necessary files and libraries

# In[ ]:


insurance_filepath = "../input/insurance.csv"

insurance_data = pd.read_csv(insurance_filepath)


# In[ ]:


insurance_data.head(10)


# ### Scatter plot view

# In[ ]:


sns.scatterplot(x = insurance_data["bmi"], y= insurance_data["charges"])


# ### Scatter plot with regression line view

# In[ ]:


#Adding a regression line on the above scatter plot
sns.regplot(x = insurance_data["bmi"], y= insurance_data["charges"])


# ### Scatter plot view with Hue - color mode

# In[ ]:


sns.scatterplot(x=insurance_data["bmi"], y = insurance_data["charges"], hue = insurance_data["smoker"])


# ### LM plot view with Hue - color mode

# In[ ]:


#Adding a regression line on the above scatter plot
sns.lmplot(x="bmi", y = "charges", hue = "smoker", data = insurance_data)


# ### Swarm plot view with Hue - categorial data

# In[ ]:


#use this to extend the scatter plots for the categorial data like gender= male/female, smoker = yes/no
sns.swarmplot(x = insurance_data["smoker"], y = insurance_data["charges"])


# ## 5. Histogram with iris dataset

# In[ ]:


iris_filepath = "../input/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")
iris_data.head()


# In[ ]:


sns.distplot(a = iris_data["Petal Length (cm)"], kde =  False) 


# ### 2. Kernel Density Plot

# In[ ]:


sns.kdeplot(data = iris_data["Petal Length (cm)"], shade = True)


# ### 3. 2D Plot

# In[ ]:


sns.jointplot(x = iris_data["Petal Length (cm)"], y = iris_data["Petal Width (cm)"], kind = "kde")


# ### Color coded plots

# In[ ]:


# Paths of the files to read
iris_set_filepath = "../input/iris_setosa.csv"
iris_ver_filepath = "../input/iris_versicolor.csv"
iris_vir_filepath = "../input/iris_virginica.csv"

# Read the files into variables 
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

# Print the first 5 rows of the Iris versicolor data
iris_ver_data.head()


# In[ ]:


sns.distplot(a = iris_set_data["Petal Width (cm)"], label = "Iris-setsova", kde= False)
sns.distplot(a = iris_ver_data["Petal Width (cm)"], label = "Iris-versicolor", kde= False)
sns.distplot(a = iris_vir_data["Petal Width (cm)"], label = "Iris-virginica", kde = False)

plt.legend()


# In[ ]:


#KDE plots for the above species
sns.set_style("whitegrid")
#sns.kdeplot(data = iris_data["Petal Length (cm)"], shade = True)
sns.kdeplot(data = iris_set_data["Petal Width (cm)"], label = 'Iris-setsova', shade = True)
sns.kdeplot(data = iris_ver_data["Petal Width (cm)"], label = 'Iris-versicolor', shade = True)
sns.kdeplot(data = iris_vir_data["Petal Width (cm)"], label = 'Iris-virginica', shade = True)


# ## 6. Styling with charts

# In[ ]:


# sns.lineplot

sns.set_style("whitegrid")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# If you have reached till here and if you like it then please upvote for the notebook. Also, once again I highly recommend to do the [course](https://www.kaggle.com/learn/data-visualization). 
