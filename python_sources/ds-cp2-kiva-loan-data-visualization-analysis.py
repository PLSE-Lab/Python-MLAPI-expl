#!/usr/bin/env python
# coding: utf-8

# In[48]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# # Visualizing Loans Awarded by Kiva
# 
# In this project you'll visualize insights using a dataset from <a href = "https://www.kaggle.com/fkosmowski/kivadhsv1" target = "_blank">Kaggle</a>. The dataset contains information about loans awarded by the non-profit <a href = "https://www.kiva.org/" target = "_blank">Kiva</a>. 
# 
# Using Seaborn, you'll explore the average loan amount by country using aggregated bar charts. You'll also visualize the distribution of loan amounts by project type and gender using box plots and violin plots.
# 
# Some of the steps below will have  hints that you can access if you need them. Hints will look like this:
# <br>
# <br>
# <details>
# <summary>Hint (click me)</summary>
# <br>
# I'm a hint!
# <br>
# </details>

# **A Note On `plt.show()`:** You may be used to displaying your plots using the code `plt.show()`. This IPython Jupyter notebook removes the necessity of calling `plt.show()` after each plot. You should be able to render your Seaborn plots simply by running the cell with the code for your plot. If you have issues rendering your plot you can try adding `plt.show()` to a cell.

# ## Step 1: Import Necessary Python Modules
# Import the modules that you'll be using in this project:
# - `from matplotlib import pyplot as plt`
# - `import pandas as pd`
# - `import seaborn as sns`

# In[49]:


from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')


# ## Step 2: Ingest The Data
# Load **kiva_data.csv** into a DataFrame called `df`. Then, quickly inspect the DataFrame using `.head()`.
# <br>
# <br>
# <details>
# <summary>Hint</summary>
# <br>
# Use `pd.read_csv()`.
# <br>
# </details>

# In[50]:


df = pd.read_csv("../input/kiva_data.csv")
print(df.head())


# ## Step 3: Examine The Data
# 
# If you would like, you can examine the raw CSV file on your local machine. You can find **kiva_data.csv** in the project download folder.
# 
# ### Overview of the dataset:
# 
# Each entry (row) in the dataset represents a loan that Kiva awarded to a particular project. The `loan_amount` column shows the amount (in U.S. dollars) awarded to the project. The `activity` column has the category type that the project falls under. The `country` column is the country where the project is located. The `gender` column represents the gender of the primary person who applied for the loan. 
# 
# 
# Print the first 25 rows of `df` using `.head()`

# In[51]:


print(df.head(25))


# ## Step 4: Bar Charts

# Create a bar plot using Seaborn to visualize the average size of Kiva loans given to projects, by country.
# 
# We've set up the figure you'll use to plot your bar plot on. The `f` variable gives us access to the figure and `ax` gives us access to the axes.
# 
# Use `sns.barplot()` with the following arguments:
# 
# - `data` set to `df`
# - `x` set to `country`
# - `y` set to `loan_amount`

# In[52]:


# Creates the figure, note: you're only using this syntax so that you can modify the y-axis ticks later
f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(data = df, x = 'country', y= 'loan_amount')


# ### Adding `$` units
# 
# You can use the following code to so that the `loan_amount` ticks on the y-axis begin with a `$` (units of USD). 
# 
# ```python
# import matplotlib.ticker as mtick
# fmt = '${x:,.0f}'
# tick = mtick.StrMethodFormatter(fmt)
# ax.yaxis.set_major_formatter(tick)
# ```

# Run the code in the cell below to see the `$` in action.

# In[53]:


import matplotlib.ticker as mtick

# Creates the figure
f, ax = plt.subplots(figsize=(15, 10))

# Plot the data
sns.barplot(data=df, x="country", y = "loan_amount")

# Use part of the code above to format the y-axis ticks below this line
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)



# ## Step 5: Learn More By Using `hue` In Your Visualization

# You can visualize even more data on one bar plot by visualizing the loan amount by country, and "nesting" by gender. Add the `hue` parameter to your `sns.barplot()` and set it so that the visualization includes the nested category of gender.
# <br>
# <br>
# <details>
# <summary>Hint</summary>
# <br>
# Set `hue` equal to the column name `gender`.
# <br>
# </details>

# In[76]:


# Creates the figure, you're only using this syntax so you can modify the y-axis ticks below
f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(data=df, x="country", y="loan_amount", hue="gender")

fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)


# #### Reflection Questions

# On average, do female or male recipients receive larger loans from Kiva?

# In[55]:


print("On average, male recipients receive larger loans from Kiva.")


# Which country has the *least* disparity in loan amounts awarded by gender?

# In[56]:


print("El Salvador has the least disparity in loan amounts awarded by gender.")


# Based on the data, what kind of recommendations can you make to Kiva about the loans they give?

# In[57]:


print("(1) Male customers generally receive larger loans from Kiva. (2) Loan amounts by genders differ significantly in Pakistan, Kenya and Cambodia. (3) In terms of loan amounts, Kiva is least popular in Philippines among the five countries.")


# What actions could be taken to implement the recommendations you've made?

# In[58]:


print("(1) Offer better policies (e.g. loan amount) to female project leaders. (2) Reinforce (1) especially in Pakistan, Kenya and Cambodia. (3) Offer better policies (e.g. loan amount) to applicants in Philippines.")


# ## Step 6: Styling
# 
# 
# Set a different color palette using `sns.set_palette()`. You can use any of the Color Brewer qualitative color palettes:
# 
# - Set1
# - Set2
# - Set3
# - Pastel1
# - Pastel2
# - Dark2
# - Accent
# 
# You can read more about <a href = "https://seaborn.pydata.org/tutorial/color_palettes.html#qualitative-color-palettes" target = "_blank">qualitative color palettes in the Seaborn documentation.</a>
# 
# 
# Set the plot background style using `sns.set_style()`. You can experiment with:
# - whitegrid
# - darkgrid
# - white
# - dark
# 
# Set the title using `ax.set_title("")`.
# 

# In[75]:


# Set color palette
sns.set_palette("Accent")

# Set style
sns.set_style("dark")

# Create figure and axes (no need to use the previous syntax, as the y-label ticks aren't going to be formatted)
plt.figure(figsize=(25, 15))

# Add a title
plt.title('')

# Use Seaborn to create the bar plot

sns.barplot(data=df, x="country", y="loan_amount", hue="gender")


# ## Step 7: Box Plots With Kiva Data
# 
# So far you have visualized the average size of loans by country using bar charts; now you are going to make a box plot to compare the distribution of loans by country.

# We have set up a figure for you to plot on. Use `sns.boxplot()` to compare the distribution of loan amounts by country for the Kiva dataset.   
# 
# `sns.boxplot()` can be passed the same parameters as `sns.barplot()`.
# 
# **Optional:** You may set a new color palette if you would like to continue using `sns.set_palette()`.
# 

# In[74]:


plt.figure(figsize=(16, 10))
sns.boxplot(data=df, x="country", y="loan_amount")


# #### Reflection Questions
# 
# Which country's box has the widest distribution?

# In[61]:


print("Kenya")


# In which country would you be most likely to receive the largest loan amount?

# In[62]:


print("Cambodia")


# ## Step 8: Box Plot by Activity

# Instead of visualizing the loan amount by *country*, use `sns.boxplot()` to plot the loan amount by *activity*.
# <br>
# <br>
# <details>
# <summary>Hint</summary>
# <br>
# You can use the same code as the box plot above, but the `x` parameter should be set to `"activity"`.
# <br>
# </details>
#  
# **Optional:** Set a different plot style and color palette to best visualize this data.

# In[73]:


plt.figure(figsize=(16, 10))
sns.boxplot(data=df, x="activity", y="loan_amount")


# #### Reflection Questions

# What does this visualization reveal that previous ones did not?

# In[64]:


print("Recipients with projects under Farming category is most likely to receive the largest amount of loan. Recipients with projects under General Store category is most likely to receive the smallest amount of loan.")


# ## Step 9: Violin Plots

# You can use nearly identical syntax (as you have used for box plots) to create violin plots. Take this line of code from above:
# 
# ```python
# sns.boxplot(data=df, x="activity", y="loan_amount")
# ```
# 
# To visualize the distribution of the exact same data as a violin plot you could pass the same parameters to `sns.violinplot()` instead of `sns.boxplot()`.
# 
# Change the code in the cell below so that the data is plotted as a violin plot instead of a barplot.

# In[72]:


plt.figure(figsize=(16, 10))

sns.violinplot(data=df, x="activity", y="loan_amount")


# ### Create a violin plot that visualizes the distribution of loan amount by country.
# Previously, you created a violin plot and plotted the data by _activity_. This time, create a violin plot that plots the data by _country_.
# 
# <br>
# <details>
# <summary>Hint</summary>
# <br>
# Change the value of the `x` argument passed into the `violinplot()` function.
# <br>
# </details>

# In[70]:


plt.figure(figsize=(16, 10))
sns.violinplot(data=df, x="country", y="loan_amount")


# ## Step 10: Split Violin Plots

# Use the `hue` and `split` parameters with `sns.violinplot()` to visualize the distribution of loan amount by country, split by gender. 
# 
# <br>
# <details>
# <summary>Hint</summary>
# <br>
# The argument `hue` should be set to `"gender"` and `split` should equal `True`. 
# <br>
# </details>

# In[71]:


# Some styling (feel free to modify)
sns.set_palette("Spectral")
plt.figure(figsize=(18, 12))
sns.violinplot(data=df, x="country", y="loan_amount", hue="gender", split=True, palette='RdBu')


# #### Reflection Questions

# What does this visualization reveal about the distribution of loan amounts within countries by gender?

# In[68]:


print("In Pakistan and Cambodia, the average loan amount received by females is much less than the average loan amount received by males. In El Salvador, there is a fairly equal parity of loan amounts between genders.")


# ## You're done! Congratulations!
# 
# You used Seaborn to visualize insights using a dataset from Kaggle. You explored the average loan amount by country using aggregated bar charts, box plots, and violin plots. You also nested the data by gender, allowing you to draw additional insights from your charts. Congratulations!
# 
# ### How do you feel?

# In[69]:


print("Great!")

