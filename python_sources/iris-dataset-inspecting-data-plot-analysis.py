#!/usr/bin/env python
# coding: utf-8

# # Iris Dataset: Inspecting data & creating plots

# **INTRO**
# 
# This week we're going to use Python to explore a dataset. We'll create outputs and plots to look for relationships within the data.
# 
# Run the following code to begin.

# In[ ]:


# Importing the libraries
import pandas as pd # Import the Pandas package library. This package is for manipulation of datasets.
import matplotlib.pyplot as plt #Import a plotting package
import seaborn as sns #Import another plotting package
get_ipython().run_line_magic('matplotlib', 'inline')
print("Complete")


# **REFERENCES**
# 
# Here's the documentation for each of the packages we'll use. Reference these whenever you get stuck. You can also search google for examples of how the code can be used to achieve different results. 
# * Seaborn, a plotting and graphing library: http://seaborn.pydata.org/api.html
# * Matplotlib, a plotting and graphing library: https://matplotlib.org/3.1.1/api/pyplot_summary.html
# * Pandas, for manipulation, Dataframes and Series: https://pandas.pydata.org/pandas-docs/stable/reference/index.html
# 
# Let's import the data. Run the following code.

# In[ ]:


dataset = pd.read_csv('../input/iris-flower-dataset/IRIS.csv') #import the data using Pandas. 

# Note: Since we imported the data strainght into Pandas, every time we work with the data it will by default be in
# the Pandas library (unless otherwise noted). 

from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://d2jmvrsizmvf4x.cloudfront.net/4TkK0t2cSGasYW8OWGJR_Vegetative%2B%28sterile%29%2B-%2Bconsist%2Bof%2Bpetals%2Band%2Bsepals.jpg", width=800)


# THE IRIS DATASET
# 
# The data we're working with is the petal and sepal lengths and widths of a few different species of iris flowers. Check out the graphic above to figure out what a petal or sepal is. 
# 
# We ultimately want to buid a model That will predict the species of iris given the sepal and petal measurements. 
# 
# The data is located here https://www.kaggle.com/arshid/iris-flower-dataset. Also check that link about the history of that dataset and how it was collected.
# 
# INSPECTING THE DATA
# 
# **pandas.head()
# **
# * We can look at a few lines of the data using .head(). The column names are listed across the top.

# In[ ]:


# By default, only the first 5 lines of the data are displayed with .head(). 

dataset.head()

# You can change the number of lines .head() displays by entering the number of rows you want to see inside the ()


# **pandas.describe()
# **
# * .describe() will calculate and output descriptive statistics for quantitative data in dataset. Notice that the species column is qualittive data and is thus not included in the describe() output.

# In[ ]:


dataset.describe()


# **pandas.info() 
# **
# * infor() tells us how many data points we have for each factor. We can use this to check if any data is missing. and what datatype is in each column.

# In[ ]:


dataset.info()


# In[ ]:


##### READING THE OUTPUT #######
# RangeIndex: The number of rows
# float64: The datatype for the first four columns. These are numbers that hold a decimal value
# object: The datatype of the species column. These are Strings that tell the species of the iris.
# Notice in the readout that each column inludes 150 non-null data points. This means we have a complete dataset.


# **pandas.value_counts()
# **
# * .value_counts() returns the frequency of data values. By specifying dataset.species.value_counts(), we only look at counts within the species column.

# In[ ]:


dataset.species.value_counts()


# WHAT WE KNOW SO FAR
# 
# From the outputs above, we can see that the dataset has info on 150 flowers. Each row contains the petal width, petal length, sepal width, sepal length, and species name of an iris flower. We know we have 50 datapoints on each of the three species.
# 
# The goal is to use the measurements of each flower to predict its species. To have an idea of how we might do that, let's look at some plots.
# 
# PLOTS
# 
# **pd.boxplot()**
# 
# * .boxplot() creates boxplots for the quantitative data. This allows to see the distribution of the data, compare between columns. Look for medians, spread, ranges, maxes and mins. 

# In[ ]:


dataset.boxplot()


# **panda.plot.scatter()**
# 
# We can do some basic plats within Pandas.

# In[ ]:


dataset.plot.scatter(x='sepal_length', y = 'sepal_width')


# If we want to see how the measurements vary by species, we'll need a better plot than that. 
# 
# **seaborn.FacetGrid**
# 
# 
# The seaborn plots allow us to color the dots in the scatterplot according to species.
# * The code below creates a scatterplot of sepal_length vs sepal_width. 
# * By setting hue="species", we're telling Seaborn to create different color dots according to species.
# * Since Seaborn is assigning colors to each species, we need a Legend so we'll know what color goes with each species.
# * height=6 sets the size of the output plot. Try changing the 6 to a different number and see how the size changes.
# * The output plots are image files that could be saved to your computer, sent in an email, or added to Word documents.

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(dataset, hue="species", height=6)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend();


# In[ ]:


#The blue species shows good seperation in this plot, but the green and yellow varieties are mixed up.


# matplotlib.pyplot.histogram()
# 
# * We can create a histogram of the data for each column to see if there's any seperation within each factor.
# * x="petal_length" tells Python to only look at the petal_length column.
# *  bins=30 sets the number of bars in the histogram to 30. Try changing that to a smaller number and see what happens.

# In[ ]:


plt.title('Histogram of Petal Lengths')
plt.hist(x="petal_length", bins=30, data=dataset)


# In[ ]:


# This histogram shows clear seperation in the petal_length column. 
# Which species do you think the petal lengths on the left side of the plot belong to?


# Now create histograms for the other three quantitative data columns: petal_width, sepal_length, and sepal_width.

# In[ ]:


#Change the histogram name by using plt.title()
plt.title('Enter new title here')

#Enter petal_widths in quotations after x=
plt.hist(x="name of column", bins=30, data=dataset)


# In[ ]:


# Copy and paste your histogram code from above and change the code to display a histogram for sepal lengths
#Code goes here:


# In[ ]:


# Copy and paste your histogram code from above and change the code to display a histogram for sepal widths
# Code goes here:


# Do any of the boxplots suggest seperation in the data for all three species?
# 
# seaborn.pairplot()
# 
# * We can use sns.pairplot() to create plots of all the factors combined pairwise.

# In[ ]:


sns.set_style("whitegrid");
sns.pairplot(dataset, hue="species", height=3);

##### READING THE PLOT #####
# Since the colors represent iris speices, we're looking for a plot that shows the best seperation.
# The measurements that show the best seperation will be used later to build our prediction models.


# In[ ]:


# Petal_width and petal_length appear to have good seperation. Let's create a plot of just those two factors.


# Copy and paste the sns.FacetGrid() code from above and change it to plot petal_length with petal_width.

# In[ ]:


# FacetGrid() code goes here:


# In[ ]:


# There appears to be a little overlap in green and yellow here, but the colors are more seperated than the original plot


# Seborn.boxplot()
# 
# Let's use Seaborn's boxplots to inspect distributions of the measurements by species. 
# 
# Run the code below to see the boxplot of peal lengths by species. Then, copy and paste the code to create the other three boxplots.

# In[ ]:


sns.boxplot(x="species", y="petal_length", data=dataset)


# In[ ]:


# Create petal width boxplot here:


# In[ ]:


# Create sepal length boxplot here:


# In[ ]:


# Create sepal width boxplot here:


# Based on the plots we've created here, what column do you think we should use to predict species?
# 
# In the next lesson, we'll 
