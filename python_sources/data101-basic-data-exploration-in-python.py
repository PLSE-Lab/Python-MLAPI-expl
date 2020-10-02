#!/usr/bin/env python
# coding: utf-8

# 
# ![banner](https://i.imgur.com/vmCCvUB.png "https://www.notion.so/sterlingdatascience")
# ## Data Science MSc - DATA101: Principles of Data Science
# # DATA101: Basic Data Exploration Lecture Notes
# 
# ## About
# 
# This is the code notebook for the "Basic Data Exploration Lecture Notes" of Sterling Osborne's Data Science MSc online course found here:
# 
# 
# ### Contents:
# 1. Basic Libraries and Importing Data
#     - Pandas and Numpy
#     - Importing Data
# 2. Selecting Data
#     - Indexing
#     - Conditionally Selecting Data
# 3. Summarising and Cleaning Data
#     - Summarising Statistics
#     - Missing values
#     - Null values
# 4. Initial Visual Analysis
#     - Matplotlib
#     - Seaborn
# 5. Conclusion
# 
# Code Notebook by Sterling Osborne
# 
# October 2019
# 
# https://twitter.com/DataOsborne
# 
# ---
# 

# # 1. Basic Libraries and Importing Data
# 
# Python on its own is fairly limited, particularly for Data Science, but instead thrives due to the open source packages available. Many of the main packages are included when you downloaded Anaconda and Jupyter and therefore all we need to do is call the ones we want at the start of each Notebook. 
# 
# 
# For now, we introduce three main packages:
# 
# - Pandas - a library for data analysis [https://pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/)
# - NumPy - a library for improved calculations [https://numpy.org/](https://numpy.org/)
# - Matplotlib - the primary plotting library for Python [https://matplotlib.org/](https://matplotlib.org/)
# 
# To call these, we simply include an import cell at the start of the Notebook as follows:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Now that we have our basic packages available, we can begin importing and exploring the data. To do this, we first use Panda's read_csv function and name our Iris dataset with a suitable Variable name. 
# 
# Lastly, we can preview the first few rows of the data by calling the '.head()' function.

# In[ ]:


# LOAD DATA FROM LOCAL DIRECTORY
#iris_data = pd.read_csv(r'C:\Users\Phil\Sync\Data Science MSc\Data\IRIS.csv')

# LOAD DATA FOR KAGGLE NOTEBOOK
iris_data = pd.read_csv('../input/iris/Iris.csv')


# In[ ]:


# Top 5 rows
iris_data.head()


# In[ ]:


# Top k rows
iris_data.head(2)


# In[ ]:


# Last 5 rows
iris_data.tail()


# ### KAGGLE DATA VARIATIONS
# 
# Because data can be uploaded to Kaggle by anyone, the data are often editied slightly from the original source. This could simply be different column names as with the Iris dataset here but in some cases a significant amount of pre-processing may have been applied before uploading the dataset. Although useful, it is important to know how the original source data has been changed and verified before using the datasets available on Kaggle and elsewhere online. 
# 
# #### Change column names for consistency with dataset downloaded directly from UCI website.

# In[ ]:


# 1. ID column not in original source so we first select only those that match with the original source
iris_data = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']]
# 2. The names of each column are slightly different so we re-name these to match the original source
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width','species']


# In[ ]:


iris_data.head()


# #### Data Types
# 
# We can use the Pandas '.dtypes' function to find the data type of each column. These are specific computer codes that relate to whether the data is a number, string or datetime.
# 
# In our case, we have four 'float64' columns, that are simply floats or decimal values, and one 'object' type which is simply a text column.
# 
# More details on these can be found in the Numpy descriptions:
# 
# https://docs.scipy.org/doc/numpy/user/basics.types.html
# 
# It is important to ensure that the data is in the correct format and the format can be changed for a column using the Pandas '.astype()' function:
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html

# In[ ]:


iris_data.dtypes


# ---
# # 2. Selecting Data
# 
# 
# ### Indexing
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html

# #### Indexing Columns
# 
# The data are in the format of Pandas dataframes, therefore to reference a column we can simply refer to its name.
# 
# This can either be done inside square brackets of by referencing the column as an attribute. The former is typically better as it means we can easily extend this for multiple columns as shown.

# In[ ]:


# Top 5 rows of 'sepal length' column
iris_data['sepal_length'].head()


# In[ ]:


# Top 5 rows of 'sepal length' column
iris_data.sepal_length.head()


# In[ ]:


# Top 5 rows of 'sepal length' column
iris_data.sepal_length.head()


# #### Indexing Rows
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# 
# Likewise, to extract a row we can refer to the row's id where in Pandas where the first row starts at 0
# 

# In[ ]:


# First row
iris_data.iloc[0]


# In[ ]:


# Rows between 3rd and 7th 
iris_data.iloc[2:6]


# Alternatively, we can reference columns by their id number, unlike rows this is not clearly shown but they can be easily counted.
# 
# To do this we use the ".iloc[row_index , col_index ]" function.
# 
# If we want all rows or columns we simply use a colon ":" instead of a number.

# In[ ]:


# First row and column element
iris_data.iloc[0,0]


# In[ ]:


# All rows and columns - top 5 rows only shown
iris_data.iloc[:,:].head()


# In[ ]:


# Rows between 3rd and 7th and all columns
iris_data.iloc[2:6,:]


# In[ ]:


# All rows and columns between 3rd and 5th - top 5 rows only shown
iris_data.iloc[:,2:5].head()


# In[ ]:


# 1st and 2nd row and 4th and 5th column
iris_data.iloc[[0,1],[3,4]]


# In[ ]:


# Rows between the 3rd and 7th and the 4th and 5th column
iris_data.iloc[2:6,[3,4]]


# In[ ]:


# 1st and 2nd rows and the 3rd and 5th
iris_data.iloc[[0,1],2:5]


# In[ ]:


# Rows between the 3rd and 7th and columns between the 3rd and 5th
iris_data.iloc[0:5,2:5]


# ### Conditionally Selecting Data Elements and Subsetting
# 
# #### Conditionally Selecting Categorical Features (String based)
# 
# We can utilise the column and row reference to subset the data as required. 
# 
# Before we are able to subset rows, it helps to know which values can be taken in categorical features. In the Iris dataset we have one categorial feature, the 'species' column and we can find the unique values in this column via Pandas '.unique()' function.
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html
# 
# 

# In[ ]:


iris_data['species'].unique()


# In[ ]:


# Select Iris-setosa data - top 5 rows only shown
iris_data[iris_data['species']=='Iris-setosa'].head()


# In[ ]:


# Select Iris-versicolor data - top 5 rows only shown
iris_data[iris_data['species']=='Iris-versicolor'].head()


# In[ ]:


# Select data with sepal_length EQUAL TO 4.9
iris_data[iris_data['sepal_length']==4.9]


# In[ ]:


# Select data with sepal_length LESS THAN 4.5
iris_data[iris_data['sepal_length']<4.5]


# In[ ]:


# Select data with sepal_length LESS THAN OR EQUAL TO 4.5
iris_data[iris_data['sepal_length']<=4.5]


# In[ ]:


# Select data with sepal_length GREATER THAN 7.4
iris_data[iris_data['sepal_length']>7.4]


# In[ ]:


# Select data with sepal_length GREATER THAN OR EQUAL TO 7.5
iris_data[iris_data['sepal_length']>=7.4]


# #### Complex Conditional Selections
# 
# To do this, we use the following functions where we use the "&" to denote AND and "|" to denote OR in subsetting notation.
# 

# In[ ]:


# Select Iris-setosa OR Iris-versicolor data - top 5 rows only shown
iris_data[(iris_data['species']=='Iris-setosa') | (iris_data['species']=='Iris-versicolor')].head()


# In[ ]:


# Select Iris-setosa AND Iris-versicolor data - no rows have both so none are returned
iris_data[(iris_data['species']=='Iris-setosa') & (iris_data['species']=='Iris-versicolor')].head()


# In[ ]:


# Select Iris-setosa AND Sepal Length EQUAL TO 4.9 
iris_data[(iris_data['species']=='Iris-versicolor') & (iris_data['sepal_length']==4.9)]


# In[ ]:


# Select Sepal Length EQUAL TO 4.8 OR Sepal Length EQUAL TO 4.9 
iris_data[(iris_data['sepal_length']==4.8) | (iris_data['sepal_length']==4.9)]


# In[ ]:


# Select Sepal Length EQUAL TO 4.8 OR Sepal Width EQUAL TO 3.0
iris_data[(iris_data['sepal_length']==4.8) & (iris_data['sepal_width']==3.0)]


# In[ ]:


# Select [Sepal Length EQUAL TO 4.8 OR Sepal Length EQUAL TO 4.9] AND Iris-setosa
iris_data[((iris_data['sepal_length']==4.8) | (iris_data['sepal_length']==4.9)) & (iris_data['species']=='Iris-setosa')]


# #### Subsetting Data
# 
# So far, each of the operations simply select and show the data based on the given conditions. We have not actually 'saved' any of these selections.
# 
# To do this, we can create a new variable name data based on the conditions to create a subset dataset.
# 

# In[ ]:


# Select Iris-setosa data - Save to new variable 'iris_data_setosa'
iris_data_setosa = iris_data[iris_data['species']=='Iris-setosa']
iris_data_setosa.head()


# --- 
# # 2. Summarising and Cleaning Data
# 
# We have already utilised the '.unique()' Pandas function for finding the values in a categorical feature, we can use the '.describe()' function to provide an overview of the other continuous features.
# 
# This returns the summary statistics:
# 
# - count: the number of elements
# - mean
# - std: the standard deviation
# - min
# - 25%: lower quartile
# - 50%: median
# - 75%: upper quartile
# - max

# In[ ]:


iris_data.describe()


# In[ ]:


iris_data['species'].unique()


# Another useful summary, particuarly useful for datasets with lots of features is the "list(data)" function that returns all of the column names.
# 
# This is also particuarly useful as it returns a list of strings for the column names that can be used later as we will show. 

# In[ ]:


list(iris_data)


# ---
# 
# # 3. Initial Visual Analysis with Matplotlib
# 
# We will follow the basic Matplotlib tutorial to create some basic plots to explore the data.
# 
# https://matplotlib.org/3.1.1/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
# 
# The important features of plots that should alway be included are:
# 
# - The title
# - x and y labels
# - Clear colours and symbols
# - Limited number of data points
# 
# 
# We can also utilise the formatting options for the marker types:
# 
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
# 
# #### Markers
# 
# character - description
# 
#     '.'	point marker
#     ','	pixel marker
#     'o'	circle marker
#     'v'	triangle_down marker
#     '^'	triangle_up marker
#     '<'	triangle_left marker
#     '>'	triangle_right marker
#     '1'	tri_down marker
#     '2'	tri_up marker
#     '3'	tri_left marker
#     '4'	tri_right marker
#     's'	square marker
#     'p'	pentagon marker
#     '*'	star marker
#     'h'	hexagon1 marker
#     'H'	hexagon2 marker
#     '+'	plus marker
#     'x'	x marker
#     'D'	diamond marker
#     'd'	thin_diamond marker
#     '|'	vline marker
#     '_'	hline marker
#     
#     
# #### Line Styles
# 
# character - description
# 
#     '-'	solid line style
#     '--'	dashed line style
#     '-.'	dash-dot line style
#     ':'	dotted line style
# 
# 
# Example format strings:
# 
#     'b'    # blue markers with default shape
#     'or'   # red circles
#     '-g'   # green solid line
#     '--'   # dashed line with default color
#     '^k:'  # black triangle_up markers connected by a dotted line
# 
# #### Colors
# 
# The supported color abbreviations are the single letter codes
# 
# character - color
# 
#     'b'	blue
#     'g'	green
#     'r'	red
#     'c'	cyan
#     'm'	magenta
#     'y'	yellow
#     'k'	black
#     'w'	white
# 
# and the 'CN' colors that index into the default property cycle.
# 
# If the color is the only part of the format string, you can additionally use any matplotlib.colors spec, e.g. full names ('green') or hex strings ('#008000').

# In[ ]:


# Basic Matplotlib plot with manually defined data
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

# Visual elements
plt.title('Basic Plot Example')
plt.ylabel('y-label')
plt.xlabel('x-label')

plt.show()


# In[ ]:


# Basic Matplotlib plot with manually defined data
# Vary the linestyle
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], '--', linewidth=5)

# Visual elements
plt.title('Basic Plot Example')
plt.ylabel('y-label')
plt.xlabel('x-label')

plt.show()


# In[ ]:


# Basic Matplotlib plot with manually defined data
# Vary the colour
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')

# Visual elements
plt.title('Basic Plot Example')
plt.ylabel('y-label')
plt.xlabel('x-label')

plt.show()


# In[ ]:


# Basic Matplotlib plot with manually defined data
# Vary the marker type
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r.')

# Visual elements
plt.title('Basic Plot Example')
plt.ylabel('y-label')
plt.xlabel('x-label')

plt.show()


# In[ ]:


# Basic Matplotlib plot with manually defined data
# Vary the marker type
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r.', markersize = 20)

# Visual elements
plt.title('Basic Plot Example')
plt.ylabel('y-label')
plt.xlabel('x-label')

plt.show()


# #### We may now plot the features from our Iris dataset

# In[ ]:


# Remind ourselves of the data with the .head() preview
iris_data.head()


# In[ ]:


# Basic Matplotlib plot for the Iris data
plt.plot(iris_data['sepal_length'], iris_data['sepal_width'], 'b.', markersize = 10)

# Visual elements
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.show()


# In[ ]:


# Basic Matplotlib plot for the Iris data
plt.plot(iris_data['petal_length'], iris_data['petal_width'], 'b.', markersize = 10)

# Visual elements
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.show()


# ### The Seaborn Plotting Library
# 
# It appears that there is a strong correlation between the Petal Length and Petal Width though it is less clear in the Sepal Length/Width comparison.
# 
# However, there seems to be some cluster groups and it may be worth investigating whether this relates to the species type. Matplotlib does not have an easy way to colour data points by another column. Instead, we may use the Seaborn package which is an extension of the Matplotlib package and has some additional aesthetic features.
# 
# https://seaborn.pydata.org/
# 
# https://seaborn.pydata.org/generated/seaborn.scatterplot.html

# In[ ]:


import seaborn as sns


# In[ ]:


# Basic Seaborn scatter plot with markers coloured by the 'Species' feature
ax = sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris_data)

# Visual elements
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.show()


# Set the Seaborn aesthetic style and increase the figure size
# 
# https://seaborn.pydata.org/tutorial/aesthetics.html

# In[ ]:


sns.set_style("white")


# In[ ]:


# Set figure size
plt.figure(figsize=(12,8))

# Basic Seaborn scatter plot with markers coloured by the 'Species' feature
ax = sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris_data, s=50)

# Visual elements
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.show()


# With the data points coloured by species, we note that correlations become clearer in certain groups. In particular, the Setosa group (blue) have a clear and strong correlation.

# ---
# 
# # 4. Conclusion
# 
# We have introduced the basics of importing and exploring data in Python and should enable you to get starting with analysing data in more detail.
