#!/usr/bin/env python
# coding: utf-8

# # Welcome!
# 
# This is a notebook. It contains a mix of markdown (a plain-text language for documentation) and python code.
# 
# Each notebook consists of multiple **cells**. This is a markdown cell and it contains documentation. The cell below this one is a cell that contains python code. You can click anywhere within a cell to edit it.

# In[ ]:


print('Ready to go!')


# Try running the cell above this one. Click anywhere in the cell and press the big blue arrow to the left of the cell to run it.
# 
# You should see the output it produces afterwords.
# 
# When you are ready, go ahead and run the cell below to load up the dataset we will use.

# In[ ]:


import pandas as pd # pandas, our library for representing tabular data
import matplotlib.pyplot as plt # matplotlib, our library for data viz
plt.rc('figure', figsize=(12, 8)) # default figure size
import seaborn as sns # additional data viz library

# load our data
df = pd.read_csv('/kaggle/input/lemonade_clean.csv')
df


# Notice that the output of the cell above is a table. This table represents the data we have now loaded into python.
# 
# We've stored our data in the `df` variable (short for dataframe). We can now explore the various *features* (aka *variables*, the columns in our table) of our data.
# 
# `df` refers to our whole dataset. To reference individual features, we use square brackets and the name of the column like this:
# 
# ```python
# df['sales']
# ```
# 
# Run the cell below to see the data for the `sales` column.

# In[ ]:


df['sales']


# To view summary statistics for a feature, we can use `.describe` like this:

# In[ ]:


df['sales'].describe()


# Now let's make some visualizations!
# 
# - histogram: `plt.hist(df['feature'])`
# - boxplot: `plt.boxplot(df['feature'])`
# - barplot `sns.barplot(data=df, y='numeric_variable', x='category')`
# - scatterplot `plt.scatter(df['feature1'], df['feature2'])`
# 
# First we'll visualize the distribution of sales:

# In[ ]:


plt.hist(df['sales'])


# There are many different ways we can adjust our plot by passing more **arguments** to the `hist` function. Multiple arguments are separated by commas.

# In[ ]:


plt.hist(df['sales'], edgecolor='pink', color='darkred', bins=20)


# Here we have adjusted the number of bins in the histogram with the `bins=20` argument. We also added a black border to the bars with the `edgecolor='pink'` argument.
# 
# To make our boxplot:

# In[ ]:


plt.boxplot(df['sales'])


# Thus far we have been using `matplotlib` to do our plotting. Next we will demonstrate `seaborn`, which provides some more "fancy" capabilities, relative to what we've seen so far.
# 
# When using `seaborn`, we'll specify what data we want to use, which values we want on the y-axis and which ones on the x-axis.
# 
# Let's take a look at the relationship between the weather and the amount of lemonade sold:

# In[ ]:


sns.barplot(data=df, y='sales', x='rainfall')


# We can swap the x and y value to get a horizontal bar chart:

# In[ ]:


sns.barplot(data=df, y='rainfall', x='sales')


# To generate a scatter plot, we can use `plt.scatter`.

# In[ ]:


plt.scatter(data=df, y='sales', x='temperature')


# ## Exercise
# 
# Explore the features of the dataset we haven't covered in the lesson. You can press the `+ Code` button to create more code cells.
# 
# 1. What happens if you use `.describe` with a categorical variable?
# 1. Show the descriptive statistics for `temperature`, `flyers` and `price`. What do you notice?
# 1. Visualize the distribution of all of the above variables. Adjust the bin size as you see fit.
# 1. Visualize relationships in the data that we haven't yet explored. (Hint: think about the type of each variable (numeric or categorical) and choose the appropriate chart type.)
#     - Does the day of the week have an impact on how much lemonade is sold?
#     - What is the relationhip between the number of flyers distributed and amount of lemonade sold?
#     - What is the relationship between the weather and the temperature?
#     - What other questions might you ask?
# 1. Is the `price` feature useful? Does it ever change?
# 1. Make your charts prettier!
# 
#     After writing the code to produce the visualization, the chart's x-axis label can be customized by adding this code:
#     
#     ```python
#     plt.xlabel('My Custom X Label')
#     ```
#     
#     Try it out!
#     
#     How do you think you would customize the y-axis label? The title?
#     
#     The range of the y-axis can be changed like this:
#     
#     ```python
#     plt.ylim(0, 100)
#     ```
#     
#     Here 0 is the minimum and 100 is the maximum. Try it out yourself!
#     
#     How do you think you would change the x-axis range? How does this effect bar charts?
#     
# 1. Different visualization types
# 
#     - Thus far, we've used a barplot (`sns.barplot`) for comparing the numeric and categorical variables. What happens if you replace `.barplot` with `.boxplot`? `.stripplot`?
#     
#     - What happens if you swap the `x` and `y` variables?
#     
#     - Try running the code below in a new cell:
#     
#         ```python
#         sns.barplot(data=df, y='sales', x='day', hue='weather')
#         ```
#         
#         What does the `hue=` do? Try this with the other chart types you explored previously as well!
#         
# 1. Even more visualization tricks
# 
#     Try using `sns.regplot` with x and y both being numeric variables. What do you notice? What happens if you add `hue=` like you did in the previous exercise?
