#!/usr/bin/env python
# coding: utf-8

# <img src="https://ucfai.org/data-science/fa19/2019-10-03-loan-defaults/loan-defaults/banner.png">
# 
# <div class="col-12">
#     <span class="btn btn-success btn-block">
#         Meeting in-person? Have you signed in?
#     </span>
# </div>
# 
# <div class="col-12">
#     <h1> Answering the Important Question: Where's My Money? </h1>
#     <hr>
# </div>
# 
# <div style="line-height: 2em;">
#     <p>by: 
#         <strong> Steve</strong>
#         (<a href="https://github.com/causallycausal">@causallycausal</a>)
#      on 2019-10-03</p>
# </div>

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns


# # Seaborn Basics
# - *If you are already familiar with Seaborn, feel free to skip this.*

# Illustrating Seaborn through the visualization of the tips datasets. For more information, please visit the Seaborn documentation [here](https://seaborn.pydata.org/).

# In[ ]:


tips = sns.load_dataset("tips") # get the tips dataset from the internet
tips # A small dataset compared to our Lending Club dataset :)


# ## Scatter Plots 
# Remember that scatter plots plot a cloud of points representing the joint distribution of two variables. In Seaborn we can use either `sns.scatterplot()` or `sns.relplot()` and note that the default `kind` of `sns.relplot()` is scatter (think of `kind` as the type of plot you want to create). Also, `sns.relplot()` is considered a `Figure-level` interface, more on that later. 

# In[ ]:


# Here we will plot tip against total_bill 
sns.relplot(x="total_bill", y="tip", data=tips); # Note here x="" and y="" corresponds to the column names in the tips dataset 


# Lets say we want to rename our x and y-axis to Total Bill and Tip, respectively and add a title to our plot. There are two possible ways to do this: 
# 1. Modifying the axes labels by calling `plt.subplots()` and use `sns.scatterplot()` to plot the data. 
# 2. Modifying the axes labels returned when calling a figure level function in Seaborn*. 
# 
# Both ways are illustrated below, feel free to choose whatever style you prefer. Additionally, I've found the first way is helpful when you want to plot multiple plots in a grid. (Remember that Seaborn is built on Matplotlib.) 
# 
# \* With every `Figure-Level` interface, a `FacetGrid` object is returned which you can access to make edits to your plot. Learn more about `FacetGrid` [here](http://seaborn.pydata.org/generated/seaborn.FacetGrid.html), building structured multi-plot grids in Seaborn [here](https://seaborn.pydata.org/tutorial/axis_grids.html) and 

# In[ ]:


# Method 1. 
fig, ax = plt.subplots()
sns.scatterplot(x="total_bill", y="tip", data=tips)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
ax.set_title("Tip vs Total Bill");


# In[ ]:


#Method 2. 
g = sns.relplot(x="total_bill", y="tip", data=tips)
g.set_axis_labels("Total Bill", "Tip")
g.ax.set_title("Tip vs Total Bill");


# Lets add a third dimension to the plot above by coloring in the points according to the categorical variable `sex`. We can achieve this by adding the attribute `hue` to our plot.

# In[ ]:


sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);


# ## Plotting Categorical Variables 
# The figure level interface for plotting categorical variables is `sns.catplot()`. We can think of different categorical plot kinds as belonging to three different families: 
# 1. Categorical scatterplots:
#     *     `sns.stripplot()` (with `kind="strip"`; the default)
#     *     `sns.swarmplot()` (with `kind="swarm"`)
# 
# 2. Categorical distribution plots:
#     
#     *     `sns.boxplot()` (with `kind="box"`)
#     *     `sns.violinplot()` (with `kind="violin"`)
#     *     `sns.boxenplot()` (with `kind="boxen"`)
# 
# 3. Categorical estimate plots:
#     *      `sns.pointplot()` (with `kind="point"`)
#     *      `sns.barplot()` (with `kind="bar"`)
#     *      `sns.countplot()` (with `kind="count"`)
#  
#  Below are some examples of each categorical plot.

# In[ ]:


# Categorical scatterplots
# Here we show how to plot total_bill within each day 
# To prevent from overlapping try kind="swarm"
sns.catplot(x="day", y="total_bill", data=tips);


# In[ ]:


# Categorical distribution plots (distributions of observations within categories)
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);


# In[ ]:


# Categorical estimate plots
# Here the height of the bars represent the mean (by default) and the error bars represent a confidence interval around the estimate
sns.catplot(x="day", y="tip", hue="sex", kind="bar", data=tips);


# # Visualizing the Distribution of a Dataset
# 
# We end this mini-tutorial with an example of how to visualize a univariate distribution.

# In[ ]:


# Here we examine the univariate distribution of tip using histograms
sns.distplot(tips["tip"], kde=False);


# In[ ]:


# Here we add a kernel density estimate (KDE) on top of the histogram.
sns.distplot(tips["tip"]);


# # Exploratory Data Analysis with the Lending Club Dataset 

#  ## Missing Value Analysis

# In[ ]:


# Load in the data 
train = pd.read_csv("../input/ucfai-dsg-fa19-default/train.csv") #we could set low_memory=True but it's useful to see that column numbers for exploration
test = pd.read_csv("../input/ucfai-dsg-fa19-default/test.csv")


# In[ ]:


# Take a look at the data with describe 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Take a look at the categorical data
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Count number of columns with at least one missing value
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Percent columns that contain at least one missing value
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Count number of row with at least one missing value
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Calculate the percentage of rows that contain at least one missing value
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Count number of columns with at least one missing value
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Calculate the percentage of nulls in each column 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Find the number of columns that have over 90% nulls in them 
# YOUR CODE HERE
raise NotImplementedError()


# ## Visualizing the Distributions of Loan Amount, Funded Amount, and Funded Amount by Investors

# In[ ]:


#visualize loan_amnt, funded_amnt, and funded_amnt_inv for the train dataset 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


#visualize loan_amnt, funded_amnt, and funded_amnt_inv for the test dataset 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Plot the number of loans issued per month
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Convert "issue_d" to pandas datetime format for the train set and then plot the number of loans issued per year 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Convert "issue_d" to pandas datetime format for the test set and then plot the number of loans issued per year 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Plot the number of loans within each grade category for the train set
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Plot the number of loans within each grade category for the test set
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Plot the number of loans within each grade category with hue train. 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# Plot the number of loans within each subgrade with hue GOOD_STANDING
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


# import pandas_profiling

# train.profile_report()

