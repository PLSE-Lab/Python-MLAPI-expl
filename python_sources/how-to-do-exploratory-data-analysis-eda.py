#!/usr/bin/env python
# coding: utf-8

# When doing data analysis EDA is step 1. EDA is done to understand your data as much as possible. It make yor familiar with data distribution, data variables and relation of variables. This guide is like a pseudo-code for doing EDA. So let's get started with..
# 
# ## What is Exploratory Data Analysis
# It is the very first step in analysis of the dataset. It is the step to prepare data before applying any model. 
# 
# But before EDA, one should spend time in really **understanding the problem**, as in why one needs to do EDA. Once you are clear with the problem statement you can relate the data to solution while doing EDA.
# 
# Find the **key variable** that is required to solve the problem. For e.g., in titanic data set it is 'survived'. In house price data set it is 'Sales Price'. Then begin with univariate analysis on thsi data variable.
# 
# Find relation of key variable with other variables, this is bivariate analysis.
# 
# Once, this is done, try to handle outliers and missing values.
# 
# It can include many steps as follow:
# 
# ## Analyzing dataset?
# Look at the metadata of the variables. What variables are available. what information do they hold. what data type they have. brief stats around those data variables. We can use following commands:
# - head
# - info
# - describe
# 
# ## Analyze key component
# In this section, use statistics and visualization to analyze one (or more) of the variable from the data set. For eg, sales price. Use `df['col'].describe()` to see stats output.
# 
# Find **distribution** of the key component by plotting a **distplot()** or histogram. Here, we can see if it is normally distributed or skewed.
# 
# ## Analyze relation of Key component with other Variables
# We now analyze key component variable with respect to other numerical variables availabe in the dataset. To do this we can make **scatter plot** between two different variables. This will show us the relation between them. Linearly relation or not. Positive or negative.. etc.
# 
# To analyze categorical variables use **boxplot**.
# 
# ## Finding Correlation
# Use `.corr()` to find correlation between all variables. THen use **heatmap** to plot the correlations. Find which varables have strong correlation.
# 
# Find fields of interest, then make a zoomed heatmap to further analyze variables or remove variables which are not of use.
# 
# Once we ahve found variables that have good relation with key variable we can create a **pairplot** of the columns of interest to compare how they relate with each other.
# 
# Now that we have found data variables that make sense and how they are related, we can think about further cleaning the data, like handling missing values and outliers.
# 
# ## Handling Missing Values
# We can use `isnull()` to find missing values. Find total number of missing values and % of missing values for each column.
# Consider how much impact missing values will have on analysis. If the columns having missing values are not the ones of our interest, drop the columns. If total missing values are very small compred to number of records, simply you may drop one or few rows.
# 
# ## Handling Outliers
# We can now change our measure to standard z score with mean of 0 and standard deviation of 1. Then we can find max and min to see how much they are away from 0. Analyze if we have outliers beyond a threshhold and if we can remove the records or keep them.
# 
# ## Standardizing Variables
# If any data is not normally distributed then we can take log of that data considering it doesn't affect much. (this needs improvement in understanding).

# In[ ]:




