#!/usr/bin/env python
# coding: utf-8

# # Welcome to Python for Data Science (with Pokemon)

# This a beginner/intermediate level Python tutorial about some of the most popular python packages in data science and scientific analysis.
# 
# This notebook was prepared by [Clarence Mah][1]. Source and license info is on [GitHub][2].
# 
# Adapted from [Andrew Gele's][3]  and [Sang Han's][4] notebooks.
# 
# 
# [1]: http://github.com/ckmah
# [2]: http://github.com/ckmah/pokemon-tutorial
# [3]: https://www.kaggle.com/ndrewgele/d/abcsds/pokemon/visualizing-pok-mon-stats-with-seaborn
# [4]: https://www.kaggle.com/sanghan/d/abcsds/pokemon/first-gen-pokemon-stats

# ### Table of Contents
# - [Jupyter Notebooks](#section1)
# - [Explore Data with Packages](#section2)
# - [Data Visualization](#section3)

# <a id='section1'></a>

# # Jupyter Notebooks
# 
# Jupyter notebooks are a medium for creating documents that contain executable code, interactive visualizations, and rich text.
# 
# Run the cell below by pressing **`Ctrl+Enter`**, or **`Shift+Enter`**.

# In[ ]:


print('This is a cell.')


# Python runs just as you would expect it to with variables, loops, etc.

# In[ ]:


numbers = [1,2,3,4]


# In[ ]:


# Look at variable
numbers


# One of the most valuable tools is tab completion. Place the cursor after **`range(`** in the following cell, then try pressing **`Shift+Tab`** a couple times slowly, noting how more and more information expands each time.

# In[ ]:


range(


# Feel free to experiment with commands and buttons in the toolbar at the top of the notebook. Use notebook shortcuts by exiting Edit Mode (press **`Esc`**) and entering Command mode. To get a list of shortcuts, press keyboard shortcut **`h`**.

# <a id='section2'></a>
# <hr>

# # Explore Data with Packages

# The open-source ecosystem around Python enables people to share libraries of code as packages. These are some of the most widely used in data science and scientific computing, including the field of bioinformatics.
# 
# [**`numpy`**][1] - [**[`cheatsheet`]**][2] A library for manipulating multi-dimensional data, linear algebra, and random number functions.
# 
# [**`pandas`**][3] - [**[`cheatsheet`]**][4] A library for performing data analysis with "relational" or "labeled" data.
# 
# [**`matplotlib`**][5] - [**[`cheatsheet`]**][6] A library for plotting figures.
# 
# [**`seaborn`**][7] - [**[`docs`]**][8] A library based on `matplotlib` that simplifies plotting with improved aesthetics.
# 
# [1]: https://numpy.org/
# [2]:  https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf
# 
# [3]: http://pandas.pydata.org/
# [4]: https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf
# 
# [5]: http://matplotlib.org/
# [6]: https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf
# 
# [7]: http://seaborn.pydata.org/
# [8]: http://seaborn.pydata.org/api

# Here, we import the libraries we want to use in this notebook.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Render plots inline
get_ipython().run_line_magic('matplotlib', 'inline')


# ## `numpy`

# `Numpy` arrays can be created several different ways. Python lists can be converted directly, while `numpy` also includes some functions for generating simple arrays.

# In[ ]:


# Python list
a_list = [3,2,1]

# Create an array from the list
a = np.array(a_list)

# Create a constant array of 2s, length 3
b = np.full(3, 2, dtype='int')


# In[ ]:


# Print
a


# In[ ]:


b


# In[ ]:


np.sort(a)


# Arrays behave like vectors, where native operators will perform element-wise operations. `numpy` also includes equivalent functions.

# In[ ]:


a + b


# In[ ]:


np.add(a, b)


# Try **`subtract()`**, **`multiply()`**, and **`divide()`** with Arrays **`a`** and **`b`** and compare the results to performing the operations with native operators: **`- * /`** .

# In[ ]:


# TODO: Try subtract, multiply and divide on array `a`


# In[ ]:


np.divide(a, b)


# `numpy` also includes handy functions like **`np.mean()`**, **`np.median()`**, **`np.std()`**, etc.

# In[ ]:


np.mean(a*b)


# In[ ]:


# TODO: Try mean, median, and std on array `a`


# There are many more useful functions that will prove useful depending on the dimensionality of your data and your specific analysis.

# <a id='section2.2'></a>

# ## `pandas`

# We will be taking a look at the Pokemon Stats dataset.
# 
# Try tab completing **`pd.read`** to see pandas' read functions. Load the pokemon dataset using the appropriate one.
# 
# Note: Use the provided **`path_to_file`** variable as the path to the data.

# In[ ]:


# TODO: Use the appropriate read function from pandas.
# Hint: the data file is a .csv file
path_to_file = '../input/Pokemon.csv'

pokemon = pd.read


# This data structure that pandas stores the data in is called a [**`DataFrame`**][1]. Think of it like a matrix, but more flexible.
# 
# [1]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

# In[ ]:


# 800 rows, 13 columns
pokemon.shape


# That might be a lot to view all at once. Preview the first 5 rows of our data with header and row names.

# In[ ]:


pokemon.head()


# **`describe()`** is a great function for a quick statistical summary of the numerical variables in your dataset.

# In[ ]:


pokemon.describe()


# We can easily select columns by name and rows by index.

# In[ ]:


# Select the `Name` column
pokemon['Name']
# OR
pokemon.Name


# In[ ]:


# Select the first 5 rows
pokemon[:5]


# A practical use for this would be to select rows and columns *conditionally*. Here we want to only look at Pokemon with more than 75 `Attack` Stat.

# In[ ]:


# Returns True of False for each row (pokemon)
pokemon.Attack > 75


# In[ ]:


# Selects pokemon with > 75 attack
pokemon[pokemon.Attack > 75]


# How would you select all the `Generation 1` Pokemon?

# In[ ]:


# TODO: Select Generation 1 pokemon


# <a id='section3'></a>
# <hr>

# # Data Visualization

# We're only going to look at first generation Pokemon for the sake of simplicity.

# In[ ]:


# TODO: Use the statement you generated from the previous code cell
pokemon = ?


# ## `matplotlib` + `seaborn`

# 
# Graphs are a great way to explore data statistics. We can make a [boxplot][1] of single variable (column) using **`seaborn`**.
# 
# [1]: https://en.wikipedia.org/wiki/Box_plot

# In[ ]:


sns.boxplot(pokemon.HP);


# Even better, `seaborn` can automatically make a boxplot for each variable out of the box.

# In[ ]:


sns.boxplot(data=pokemon);


# Since we only want to look at their stats, some variables are irrelevant. Let's exclude those for now.
# 
# Note: If you are confused about a function or there are too many parameters to keep track of, remember to use tab completion for help. Put your cursor after `pokemon.drop(` and try it out.

# In[ ]:


pokemon = pokemon.drop(['#', 'Total', 'Legendary'],axis='columns')


# Oops! There's one more variable that we forgot to drop. Let's drop the `Generation` column since we only have Generation 1 in our `DataFrame`.
# 
# Don't forget to reassign the result to `pokemon` or the dropped column won't be saved.

# In[ ]:


# TODO: Drop Generation column here


# Notice that we now only have relevant columns that are considered a Stat or Pokemon Type.

# In[ ]:


pokemon.head()


# Great! Let's plot it again with only Stats.

# In[ ]:


sns.boxplot(data=pokemon);


# ### Fancier Plots

# We can compare Pokemon stats by type. This particular plot, [`swarmplot`][1] requires data transformed in a certain format. Check out pandas' [`melt`][1] for a description of the transformation.
# 
# [1]: http://seaborn.pydata.org/generated/seaborn.swarmplot.html
# [2]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html

# In[ ]:


# Transform data for swarmplot
normalized = pd.melt(
    pokemon, id_vars=["Name", "Type 1", "Type 2"], var_name="Stat")


# In[ ]:


normalized.head()


# In[ ]:


sns.swarmplot(data=normalized, x='Stat', y='value', hue='Type 1');


# That looks neat, but seems to be a little cluttered. Using a few Seaborn and Matplotlib functions, we can adjust how our plot looks.
# On each line below, we will:
# - [Make the plot larger](http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure)
# - [Adjust the y-axis](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ylim)
# - Organize the point distribution by type and make the individual points larger
# - [Move the legend out of the way](http://matplotlib.org/users/legend_guide.html#legend-location)

# In[ ]:


# Make the plot larger
plt.figure(figsize=(12, 10))

# Adjust the y-axis
plt.ylim(0, 275)

# Organize by type [split], make points larger [size]
sns.swarmplot(
    data=normalized, x='Stat', y='value', hue='Type 1', split=True, size=7);

# Move legend out of the way
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);


# Make it easier to see types...

# In[ ]:


# Set background to white
sns.set_style('whitegrid')

# Make the plot larger
plt.figure(figsize=(12, 10))

# Adjust the y-axis
plt.ylim(0, 275)

# Organize by type [split], make points larger [size]
sns.swarmplot(
    data=normalized,
    x='Stat',
    y='value',
    hue='Type 1',
    split=True,
    size=7,
    palette='Set3');

# Move legend out of the way
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);


# ## Bonus plot!
# 
# Compare Stats by types.

# In[ ]:


# Compare stats by type
figure = sns.factorplot(
    data=normalized,
    x='Type 1',
    y='value',
    col='Stat',
    col_wrap=2,
    aspect=2,
    kind='box',
    palette='Set3');

# Rotate x-axis tick labels
figure.set_xticklabels(rotation=30);


# <a id="section4"></a>

# 
