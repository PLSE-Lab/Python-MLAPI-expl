#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/ealmIaI.jpg)

# # **Introduction**
# The **<span style="color:violet">pandas</span>** library has emerged into a power house of data manipulation tasks in **<span style="color:turquoise">Python</span>** since it was developed in 2008. With its intuitive syntax and flexible data structure, it's easy to learn and enables faster data computation. The development of **<span style="color:violet">numpy</span>** and **<span style="color:violet">pandas</span>** libraries has extended **<span style="color:turquoise">Python</span>**'s multi-purpose nature to solve machine learning problems as well. The acceptance of **<span style="color:turquoise">Python</span>** language in machine learning has been phenomenal since then.
# 
# This is just one more reason underlining the need for you to learn these libraries now. As claimed 2-3 years ago by many, **<span style="color:turquoise">Python</span>** jobs have finally outnumbered **<span style="color:turquoise">R</span>** jobs.
# 
# In this tutorial, we'll learn about using **<span style="color:violet">numpy</span>** and **<span style="color:violet">pandas</span>** libraries for data manipulation from scratch. Instead of going into theory, we'll take a practical approach.
# 
# First, we'll understand the syntax and commonly used functions of the respective libraries. Later, we'll work on a real-life data set.
# 
# Note: This tutorial is best suited for people who know the basics of **<span style="color:turquoise">Python</span>**. No further knowledge is expected.
# 
# https://www.hackerearth.com/practice/machine-learning/data-manipulation-visualisation-r-python/tutorial-data-manipulation-numpy-pandas-python/tutorial/
# 
# https://plot.ly/python/basic-charts/

# ## **Table of Contents**
# 1. Six important things you should know about **<span style="color:violet">numpy</span>** and **<span style="color:violet">pandas</span>**
# 2. Starting with **<span style="color:violet">numpy</span>**
# 3. Starting with **<span style="color:violet">pandas</span>**
# 4. Stating with **<span style="color:violet">plotly</span>**
# 5. Exploring an ML Data Set
# 6. Building a Random Forest Model

# ![](https://i.imgur.com/etL15rR.jpg)

# # **Six important things you should know about <span style="color:indigo">numpy</span> and <span style="color:indigo">pandas</span> **
# 
# 1. The data manipulation capabilities of **<span style="color:violet">pandas</span>** are built on top of the **<span style="color:violet">numpy</span>** library. In a way, **<span style="color:violet">numpy</span>** is a dependency of the **<span style="color:violet">pandas</span>** library.
# 2. The **<span style="color:violet">pandas</span>** library is best at handling tabular data sets comprising different variable types (integer, float, double, etc.). In addition, the **<span style="color:violet">pandas</span>** library can also be used to perform even the most naive of tasks such as loading data or doing feature engineering on time series data.
# 3. The **<span style="color:violet">numpy</span>** is most suitable for performing basic numerical computations such as **<span style="color:purple">mean</span>**, **<span style="color:purple">median</span>**, **<span style="color:purple">range</span>**, etc. Alongside, it also supports the creation of multi-dimensional arrays.
# 4. The **<span style="color:violet">numpy</span>** library can also be used to integrate **<span style="color:turquoise">C/C++</span>** and **<span style="color:turquoise">Fortran</span>** code.
# 5. Remember, **<span style="color:turquoise">Python</span>** is a zero indexing language unlike **<span style="color:turquoise">R</span>** where indexing starts at one.
# 6. The best part of learning **<span style="color:violet">pandas</span>** and **<span style="color:violet">numpy</span>** is the strong active community support you'll get from around the world.
# 
# Just to give you a flavor of the **<span style="color:violet">numpy</span>** library, we'll quickly go through its syntax structures and some important commands such as slicing, indexing, concatenation, etc. All these commands will come in handy when using **<span style="color:violet">pandas</span>** as well. Let's get started!

# ![](https://i.imgur.com/howaGlg.jpg)

# # **Starting with <span style="color:indigo">numpy</span>**

# In[ ]:


# Load the library and check its version, just to make sure we aren't using an older version.
import numpy as np
np.__version__


# In[ ]:


# Create a list comprising numbers from 0 to 9.
L = list(range(10))

[str(c) for c in L]


# In[ ]:


# Converting integers to string - this style of handling lists is known as list comprehension.
# List comprehension offers a versatile way to handle list manipulation tasks easily. We'll learn about them in future tutorials. 
# Here's an example.  

[type(item) for item in L]


# ### **Creating Arrays**
# The **<span style="color:violet">numpy</span>** arrays are homogeneous in nature, i.e., they comprise one data type (integer, float, double, etc.) unlike lists.

# In[ ]:


# Create zero arrays
np.zeros(10, dtype='int')


# In[ ]:


# Create a 3 row x 5 column matrix of ones
np.ones((3, 5), dtype=float)


# In[ ]:


# Create a matrix with a predefined value
np.full((3, 5), 1.23)


# In[ ]:


# Create an array with a set sequence
np.arange(0, 20, 2)


# In[ ]:


# Create an array of even space between the given range of values
np.linspace(0, 1, 5)


# In[ ]:


# Create a 3 x 3 array of values picked from the normal distribution with mean 0 and standard deviation 1 in a given dimension
np.random.normal(0, 1, (3, 3))


# In[ ]:


# Create a 3 x 3 array of values picked from the Binomial (n, p) distribution in a given dimension
np.random.binomial(10, 0.5, (3, 5))


# In[ ]:


# Create an identity matrix
np.eye(3)


# In[ ]:


# Set a random seed
np.random.seed(0)


# In[ ]:


x1 = np.random.randint(10, size=6) # one dimension
x2 = np.random.randint(10, size=(3, 4)) # two dimensions
x3 = np.random.randint(10, size=(3, 4, 5)) # three dimensions

print("x3 ndim:", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)


# ### **Exercise**

# In[ ]:


# Create a 4 x 4 identity matrix

# Create a 3 x 10 matrix with all values equal '2'

# Create a 2 x 3 x 2 x 4 4D-matrix and print size and shape


# ### **Array Indexing**
# The important thing to remember is that indexing in **<span style="color:turquoise">Python</span>** starts at zero.

# In[ ]:


x1 = np.array([4, 3, 4, 4, 8, 4])
x1


# In[ ]:


# Access the value at index zero
x1[0]


# In[ ]:


# Access the fifth value
x1[4]


# In[ ]:


# Access the last value
x1[-1]


# In[ ]:


# Access the second last value
x1[-2]


# In[ ]:


# In a multidimensional array, we need to specify row and column indices
x2


# In[ ]:


# 1st row and 2nd column value
x2[2, 3]


# In[ ]:


# 3rd row and last column value
x2[2, -1]


# In[ ]:


# Replace value at index [0, 0]
x2[0, 0] = 12
x2


# ### **Exercise**

# In[ ]:


# Using the x3 matrix (using index -1 when necessary)...

# Print the value at 3D-index [2, 0, 4]

# Modify the value at 3D-index [0, 3, 1] to be 100


# ### **Array Slicing**
# Now, we'll learn to access a range of elements from an array.

# In[ ]:


x = np.arange(10)
x


# In[ ]:


# From the start to the 4th position
x[:5]


# In[ ]:


# From the 4th position to the end
x[3:]


# In[ ]:


# From 4th to 7th position
x[3:7]


# In[ ]:


# Return elements with even indices
x[:: 2]


# In[ ]:


# Return elements from the 1st position to the end by step two
x[1::2]


# In[ ]:


# Reverse the array
x[::-1]


# ### **Exercise**

# In[ ]:


# Create an array of consecutive numbers from 0 to 50 

# Print every number divisible by 6

# Then reverse that array

# Print the element at the 7th index


# ### **Array Concatenation**
# Frequently, we are required to combine different arrays. So, instead of typing each of their elements manually, you can use array concatenation to handle such tasks easily.

# In[ ]:


# You can concatenate two or more arrays at once.
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = [21, 21, 21]
np.concatenate([x, y, z])


# In[ ]:


# You can also use this function to create 2-dimensional arrays.
grid = np.array([[1, 2, 3], [4, 5, 6]])
np.concatenate([grid, grid])


# In[ ]:


# Using its axis parameter, you can define a row-wise or column-wise matrix
print(np.concatenate([grid, grid], axis=0))
print()
print(np.concatenate([grid, grid], axis=1))


# Yet we used the function **<span style="color:purple">concatenate</span>** with arrays of equal dimension. But, what if you are required to combine a 2D-array with a 1D-array? In such situations **<span style="color:purple">concatenate</span>** might not be the best option to use. Instead, you can use **<span style="color:purple">vstack</span>** or **<span style="color:purple">hstack</span>** to complete the task. Let's see how!

# In[ ]:


x = np.array([3, 4, 5])
grid = np.array([[1, 2, 3], [17, 18, 19]])
np.vstack([x, grid])


# In[ ]:


# Similarly, you can merge arrays using np.hstack
z = np.array([[9], [9]])
np.hstack([grid, z])


# Also, we can split arrays based on predefined positions. Let's see how!

# In[ ]:


x = np.arange(10)
x


# In[ ]:


x1, x2, x3 = np.split(x, [3, 6])
print(x1, x2, x3)


# In[ ]:


grid = np.arange(16).reshape((4, 4))
print(grid)
print()
upper, lower = np.vsplit(grid, [3])
print(upper)
print()
print(lower)


# ### **Exercise**

# In[ ]:


# Create a 1D-array of size 4 with numbers randomly distributed from 1 to 10

# Create a 2D-array of size 2 x 4 with numbers randomly distributed from 10 to 20

# Vertically stack the arrays to obtain a 3 x 4 matrix


# In addition to the functions we learned above, there are several other mathematical functions available in the **<span style="color:violet">numpy</span>** library such as **<span style="color:purple">sum</span>**, **<span style="color:purple">divide</span>**, **<span style="color:purple">multiple</span>**, **<span style="color:purple">abs</span>**, **<span style="color:purple">power</span>**, **<span style="color:purple">mod</span>**, **<span style="color:purple">sin</span>**, **<span style="color:purple">cos</span>**, **<span style="color:purple">tan</span>**, **<span style="color:purple">log</span>**, **<span style="color:purple">var</span>**, **<span style="color:purple">min</span>**, **<span style="color:purple">max</span>**, **<span style="color:purple">mean</span>**, etc. which you can use to perform basic arithmetic calculations. Feel free to refer to **<span style="color:violet">numpy</span>** documentation for more information on such functions.
# 
# Let's move on to **<span style="color:violet">pandas</span>** now. Make sure you following each line below because it'll help you in doing data manipulation using **<span style="color:violet">pandas</span>**.

# ![](https://i.imgur.com/WvSt0DR.jpg)

# # **Let's start with <span style="color:indigo">pandas</span>**

# In[ ]:


# Load library - pd is just an alias. We use pd because it's short and literally abbreviates pandas.
# You can use any name as an alias. 
import pandas as pd


# ### **Data Frame Creation**

# In[ ]:


# Create a data frame - dictionary is used here where keys get converted to column names and values (represented as lists) to row values.
data = pd.DataFrame({'Country': ['Russia','Colombia','Chile','Equador','Nigeria'],
                    'Rank':[121,40,100,130,11]})
data


# In[ ]:


# We can do a quick analysis of any data set using:
data.describe()


# Remember that the **<span style="color:purple">describe()</span>** method computes summary statistics of integer / double variables. To get the complete information about the data set, we can use the **<span style="color:purple">info()</span>** function.

# In[ ]:


# Among other things, it shows that the data set has 5 rows and 2 columns with their respective names.
data.info()


# ### **Sorting Data Frame**

# In[ ]:


# Let's create another data frame.
data = pd.DataFrame({'group':['a', 'a', 'a', 'b','b', 'b', 'c', 'c','c'],'ounces':[4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[ ]:


# Let's sort the data frame by ounces, where inplace = True will make changes to the data
data.sort_values(by=['ounces'], ascending=True, inplace=False)


# We can sort <span style="color:orangered">data</span> by not just one column but multiple columns as well.

# In[ ]:


data.sort_values(by=['group','ounces'], ascending=[True,False], inplace=False)


# ### **Removing Duplicates**

# Often, we get data sets with duplicate rows, which is nothing but noise. Therefore, before training the model, we need to make sure we get rid of such inconsistencies in the data set. Let's see how we can remove duplicate rows.

# In[ ]:


# Create another data with duplicated rows
data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [3, 2, 1, 3, 3, 4, 4]})
data


# In[ ]:


# Sort values
data.sort_values(by='k2')


# In[ ]:


# Remove duplicates - ta daaaa!
data.drop_duplicates()


# Here, we removed duplicates based on matching row values across all columns. Alternatively, we can also remove duplicates based on a particular column. Let's remove duplicate values from the <span style="color:teal">k1</span> column.

# In[ ]:


data.drop_duplicates(subset='k1')


# Again, remember that in order to do permanent changes we need to use **<span style="color:purple">inplace=True</span>**. In the above example the data frame <span style="color:orangered">data</span> has not changed in spite of calling **<span style="color:purple">data.drop_duplicates()</span>**.

# In[ ]:


# Check data frame data
data


# In[ ]:


# Remove duplicates physically from data
data.drop_duplicates(inplace=True)
data


# ### **Exercise**

# In[ ]:


# Initialize data
data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [3, 2, 1, 3, 3, 4, 4]})

# Drop duplicates from k2 column

# Sort by k2 column in ascending order


# ### **Creating New Variables**

# Now, we will learn to categorize rows based on a predefined criteria. It happens a lot while data processing when you need to categorize a variable. For example, say we have got a column with food items and we want to create a new variable <span style="color:teal">meat_to_animal</span> based on these food items. In such situations, we will require the steps below:

# In[ ]:


data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham','nova lox'],
                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# Now, we want to create a new variable which indicates the type of animal which acts as the source of food. To do that, first we'll create a dictionary to map food to animals. Then, we'll use a map function to map the dictionary's values to the keys. Let's see how it is done.

# In[ ]:


meat_to_animal = {
'bacon': 'pig',
'pulled pork': 'pig',
'pastrami': 'cow',
'corned beef': 'cow',
'honey ham': 'pig',
'nova lox': 'salmon'
}

def meat_2_animal(series):
    if series['food'] == 'bacon':
        return 'pig'
    elif series['food'] == 'pulled pork':
        return 'pig'
    elif series['food'] == 'pastrami':
        return 'cow'
    elif series['food'] == 'corned beef':
        return 'cow'
    elif series['food'] == 'honey ham':
        return 'pig'
    else:
        return 'salmon'


# Create a new variable
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data


# In[ ]:


# Another way of doing it is to convert the food values to lower case and apply the function meat_2_animal
lower = lambda x: x.lower()
data['food'] = data['food'].apply(lower)
data['animal2'] = data.apply(meat_2_animal, axis='columns')
data


# Another way to create a new variable is by using the function **<span style="color:purple">assign()</span>**. In this tutorial, as you keep discovering new functions, you'll realize how powerful **<span style="color:violet">pandas</span>** is.

# In[ ]:


data.assign(new_variable = data['ounces'] * 10)


# Let's remove the column animal2 from our data frame.

# In[ ]:


data.drop('animal2',axis='columns',inplace=True)
data


# ### **Exercise**

# In[ ]:


data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [3, 2, 1, 3, 3, 4, 4]})
data

# Build a function to convert k1 (string) into an integer number by using the above meat_2_animal function

# Create a new column by multiplying k2 and the new column you have just created

# Drop k1 column


# ### **Handling Outliers and Missing Values**

# We frequently find missing values in our data set. A quick method for handling missing values is to fill a missing value with any random number. Besides missing values, you may find a lot of outliers in your data set, which may require to be replaced as well. Let's see how can we accomplish this task.

# In[ ]:


# The function Series from pandas is used to create pandas arrays
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data


# In[ ]:


# It is not a numpy array
type(data)


# In[ ]:


# Replace -999 with NaN values
data.replace(-999, np.nan, inplace=True)
data


# In[ ]:


# We can also replace multiple values at once
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace([-999, -1000], np.nan, inplace=True)
data


# ### **Renaming Rows and Columns**

# Now, let's learn how to rename column names and row names (axes).

# In[ ]:


data = pd.DataFrame(np.arange(12).reshape((3, 4)),index=['Ohio', 'Colorado', 'New York'],columns=['one', 'two', 'three', 'four'])
data


# In[ ]:


# Using rename function
data.rename(index = {'Ohio':'SanF'}, columns={'one':'one_p','two':'two_p'}, inplace=True)
data


# In[ ]:


# You can also use string functions
data.rename(index = str.upper, columns=str.title, inplace=True)
data


# Next, we'll learn to categorize (bin) continuous variables.

# In[ ]:


ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]


# We'll divide <span style="color:orangered">ages</span> into bins such as <span style="color:navy">19-25</span>, <span style="color:navy">26-35</span>, <span style="color:navy">36-60</span> and <span style="color:navy">61+</span>.

# In[ ]:


# Understand the output - '(' means the value is included in the bin, '[' means the value is excluded
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats


# In[ ]:


# To include the right bin value, we can do:
pd.cut(ages, bins, right=False)


# In[ ]:


# Let's check how many observations fall under each bin
pd.value_counts(cats)


# Also, we can pass a unique name to each label.

# In[ ]:


bin_names = ['Youth', 'YoungAdult', 'MiddleAge', 'Senior']
new_cats = pd.cut(ages, bins, labels=bin_names)

pd.value_counts(new_cats)


# In[ ]:


# We can also calculate their cumulative sum
pd.value_counts(new_cats).cumsum()


# ### **Exercise**

# In[ ]:


salary = [120, 222, 25, 127, 121, 93, 337, 51, 31, 85, 41, 62]
bins = [0, 42, 126, 1000]
bin_names = ['Lower Class', 'Middle Class', 'Upper Class']
# Split the salaraies into the specified bins

# Apply the bin names to the categories


# Let's proceed and learn about grouping data and creating pivots in **<span style="color:violet">pandas</span>**. It's an immensely important data analysis method which you'd probably have to use on every data set you work with.

# In[ ]:


df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})
df


# In[ ]:


# Calculate the mean of data1 column by key1
grouped = df['data1'].groupby(df['key1'])
grouped.mean()


# Now, let's see how to slice the data frame.

# In[ ]:


dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df


# In[ ]:


# Get first n rows from the data frame
df[:3]


# In[ ]:


# Slice based on date range
df['20130101':'20130104']


# In[ ]:


# Slicing based on column names
df.loc[:, ['A','B']]


# In[ ]:


# Slicing based on both row index labels and column names
df.loc['20130102': '20130103', ['A','B']]


# In[ ]:


# Slicing based on index of columns
df.iloc[3] # returns the 4th row (index is the 3rd)


# In[ ]:


# Returns a specific range of rows
df.iloc[2:4, 0:2]


# In[ ]:


# Returns specific rows and columns using lists containing columns or row indices
df.iloc[[1,5],[0,2]] 


# Similarly, we can do Boolean indexing based on column values as well. This helps in filtering a data set based on a predefined condition.

# In[ ]:


df[df.A > 1]


# In[ ]:


# We can copy the data set
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2


# In[ ]:


# Select rows based on column values
df2[df2['E'].isin(['two', 'four'])]


# In[ ]:


# Select all rows except those with two and four
df2[~df2['E'].isin(['two','four'])]


# We can also use a query method to select columns based on a criterion. Let's see how!

# In[ ]:


# List all columns where A is greater than C
df.query('A > C')


# In[ ]:


# Using OR condition
df.query('A < B | C > A')


# ### **Exercise**

# In[ ]:


iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
iris

# What is the average measurements for each species

# Return these 2 indexs: [1,5] [0,2]

# Return the sepal_length and species column

# Return rows in list: ['setosa','virginica']

# Are there any siutations that either sepal or petal widths are greater that lengths?


# Pivot tables are extremely useful in analyzing data using a customized tabular format. I think, among other things, **<span style="color:turquoise">Excel</span>** is popular because of the pivot table option. It offers a super-quick way to analyze data.

# In[ ]:


# Create a data frame
data = pd.DataFrame({'group': ['a', 'a', 'a', 'b','b', 'b', 'c', 'c', 'c'],
                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[ ]:


# Calculate means of each group
data.pivot_table(values='ounces', index='group', aggfunc=np.mean)


# In[ ]:


# Calculate count by each group
data.pivot_table(values='ounces', index='group', aggfunc='count')


# Up till now, we've become familiar with the basics of **<span style="color:violet">pandas</span>** library using toy examples. Now, we'll take up a real-life data set and use our newly gained knowledge to explore it.
# 
# https://matplotlib.org/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py
# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
# 

# ![](https://i.imgur.com/tSq1kAT.jpg)

# # **<span style="color:indigo">Plotly</span>**
# 
# The rendermode argument to supported **<span style="color:violet">Plotly Express</span>** functions can be used to enable <span style="color:turquoise">WebGL</span> rendering.
# 
# Here is an example that creates a 100,000 point scatter plot using **<span style="color:violet">Plotly Express</span>** with <span style="color:turquoise">WebGL</span> rendering enabled.

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib, json


# In[ ]:


np.random.seed(1)

N = 100000

df = pd.DataFrame(dict(x=np.random.randn(N), y=np.random.randn(N)))

fig = px.scatter(df, x="x", y="y", render_mode='webgl')

fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'))

fig.show()


# ## **Scatter plot with <span style="color:indigo">Plotly Express</span>**
# 
# **<span style="color:violet">Plotly Express</span>** is the easy-to-use, high-level interface to **<span style="color:violet">Plotly</span>**, which operates on "tidy" data.
# 
# With **<span style="color:purple">px.scatter()</span>**, each data point is represented as a marker point, which location is given by the <span style="color:blue">x</span> and <span style="color:blue">y</span> columns.

# In[ ]:


# x and y given as DataFrame columns
iris = px.data.iris() # iris is a pandas DataFrame
fig = px.scatter(iris, x="sepal_width", y="sepal_length")
fig.show()


# ### **Set size and color with column names**
# Note that color and size data are added to hover information. You can add other columns to hover data with the <span style="color:red">hover_data</span> argument of **<span style="color:purple">px.scatter()</span>**.

# In[ ]:


iris = px.data.iris()
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()


# ### **Line plot with <span style="color:indigo">Plotly Express</span>**

# In[ ]:


gapminder = px.data.gapminder().query("continent == 'Oceania'")
fig = px.line(gapminder, x='year', y='lifeExp', color='country')
fig.show()


# ## **Scatter and line plot with <span style="color:purple">go.Scatter</span>**
# 
# If **<span style="color:violet">px.scatter()</span>** does not provide a good starting point, it is possible to use the more generic **<span style="color:purple">go.Scatter</span>** function from **<span style="color:violet">plotly.graph_objects</span>**. Whereas **<span style="color:violet">plotly.express</span>** has two functions **<span style="color:purple">scatter</span>** and **<span style="color:purple">line</span>**, **<span style="color:purple">go.Scatter</span>** can be used both for plotting points (makers) or lines, depending on the value of mode. The different options of **<span style="color:purple">go.Scatter</span>** are documented in its reference page.

# ### **Simple Scatter Plot**

# In[ ]:


N = 1000
t = np.linspace(0, 10, 100)
y = np.sin(t)

fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))

fig.show()


# ### **Line and Scatter Plots**
# Use mode argument to choose between markers, lines, or a combination of both. For more options about line plots, see also the line charts notebook and the filled area plots notebook.

# In[ ]:


# Create random data with numpy
np.random.seed(1)

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='markers',
                    name='markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                    mode='lines+markers',
                    name='lines+markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y2,
                    mode='lines',
                    name='lines'))

fig.show()


# ### **Style Scatter Plots**

# In[ ]:


t = np.linspace(0, 10, 100)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, y=np.sin(t),
    name='sin',
    mode='markers',
    marker_color='rgba(152, 0, 0, .8)'
))

fig.add_trace(go.Scatter(
    x=t, y=np.cos(t),
    name='cos',
    marker_color='rgba(255, 182, 193, .9)'
))

# Set options common to all traces with fig.update_traces
fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig.update_layout(title='Styled Scatter',
                  yaxis_zeroline=False, xaxis_zeroline=False)


fig.show()


# ### **Data Labels on Hover**

# In[ ]:


data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")

fig = go.Figure(data=go.Scatter(x=data['Postal'],
                                y=data['Population'],
                                mode='markers',
                                marker_color=data['Population'],
                                text=data['State'])) # hover text goes here

fig.update_layout(title='Population of USA States')
fig.show()


# ### **Scatter with a Color Dimension**

# In[ ]:


fig = go.Figure(data=go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color=np.random.randn(500), #set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        showscale=True
    )
))

fig.show()


# ### **Dash Example**
# **<span style="color:violet">Dash</span>** is an Open Source **<span style="color:turquoise">Python</span>** library which can help you convert **<span style="color:violet">plotly</span>** figures into a reactive, web-based application. Below is a simple example of a dashboard created using **<span style="color:violet">Dash</span>**. Its source code can easily be deployed to a PaaS.

# In[ ]:


from IPython.display import IFrame
IFrame(src= "https://dash-simple-apps.plotly.host/dash-linescatterplot/", width="100%",height="750px", frameBorder="0")


# In[ ]:


from IPython.display import IFrame
IFrame(src= "https://dash-simple-apps.plotly.host/dash-linescatterplot/code", width="100%",height=500, frameBorder="0")


# ## **Bubble chart with <span style="color:indigo">Plotly Express</span>**
# 
# A bubble chart is a scatter plot in which a third dimension of the data is shown through the size of markers. For other types of scatter plot, see the line and scatter page.
# 
# We first show a bubble chart example using **<span style="color:violet">Plotly Express</span>**. **<span style="color:violet">Plotly Express</span>** is the easy-to-use, high-level interface to **<span style="color:violet">Plotly</span>**, which operates on "tidy" data. The size of markers is set from the dataframe column given as the size parameter.

# In[ ]:


gapminder = px.data.gapminder()

fig = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp",size="pop", color="continent",hover_name="country", log_x=True, size_max=60)
fig.show()


# ## **Line Plot with <span style="color:indigo">Plotly Express</span>**
# 
# With **<span style="color:purple">px.line</span>**, each data point is represented as a vertex (which location is given by the <span style="color:blue">x</span> and <span style="color:blue">y</span> columns) of a polyline mark in 2D space.

# In[ ]:


# Add data
month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']
high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]
low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]
high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]
low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]
high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]
low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]

fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=month, y=high_2014, name='High 2014',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=month, y=low_2014, name = 'Low 2014',
                         line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=month, y=high_2007, name='High 2007',
                         line=dict(color='firebrick', width=4,
                              dash='dash') # dash options include 'dash', 'dot', and 'dashdot'
))
fig.add_trace(go.Scatter(x=month, y=low_2007, name='Low 2007',
                         line = dict(color='royalblue', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=month, y=high_2000, name='High 2000',
                         line = dict(color='firebrick', width=4, dash='dot')))
fig.add_trace(go.Scatter(x=month, y=low_2000, name='Low 2000',
                         line=dict(color='royalblue', width=4, dash='dot')))

# Edit the layout
fig.update_layout(title='Average High and Low Temperatures in New York',
                   xaxis_title='Month',
                   yaxis_title='Temperature (degrees F)')


fig.show()


# ## **Filled area plot with <span style="color:indigo">Plotly Express</span>**
# 
# The function **<span style="color:indigo">px.area</span>** creates a stacked area plot. Each filled area corresponds to one value of the column given by the <span style="color:red">line_group</span> parameter.

# In[ ]:


gapminder = px.data.gapminder()
fig = px.area(gapminder, x="year", y="pop", color="continent",line_group="country")
fig.show()


# ## **Bar chart with <span style="color:indigo">Plotly Express</span>**
# 
# With **<span style="color:purple">px.bar</span>**, each row of the DataFrame is represented as a rectangular mark.

# In[ ]:


x=['a','b','c','d']
fig = go.Figure(go.Bar(x=x, y=[2, 5, 1, 9], name='Montreal'))
fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))
fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))

fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':['d','a','c','b']})
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Bar(
    x=[1, 2, 3, 5.5, 10],
    y=[10, 8, 6, 4, 2],
    width=[0.8, 0.8, 0.8, 3.5, 4] # customize width here
)])

fig.show()


# In[ ]:


tips = px.data.tips()
fig = px.bar(tips, x="sex", y="total_bill", color="smoker", barmode="group",
             facet_row="time", facet_col="day",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],
                              "time": ["Lunch", "Dinner"]})
fig.show()


# In[ ]:


years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
         2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]

fig = go.Figure()
fig.add_trace(go.Bar(x=years,
                y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                   350, 430, 474, 526, 488, 537, 500, 439],
                name='Rest of world',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=years,
                y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
                   299, 340, 403, 549, 499],
                name='China',
                marker_color='rgb(26, 118, 255)'
                ))

fig.update_layout(
    title='US Export of Plastic Scrap',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='USD (millions)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# ## **Horizontal Bar Chart with <span style="color:indigo">Plotly Express</span>**
# 
# For a horizontal bar char, use the function **<span style="color:purple">px.bar</span>** with <span style="color:blue">orientation='h'</span>.

# In[ ]:


fig = go.Figure(go.Bar(
            x=[20, 14, 23],
            y=['giraffes', 'orangutans', 'monkeys'],
            orientation='h'))

fig.show()


# In[ ]:


tips = px.data.tips()
fig = px.bar(tips, x="total_bill", y="sex", color='day', orientation='h',
             hover_data=["tip", "size"],
             height=400,
             title='Restaurant bills')
fig.show()


# In[ ]:


top_labels = ['Strongly<br>agree', 'Agree', 'Neutral', 'Disagree',
              'Strongly<br>disagree']

colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
          'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
          'rgba(190, 192, 213, 1)']

x_data = [[21, 30, 21, 16, 12],
          [24, 31, 19, 15, 11],
          [27, 26, 23, 11, 13],
          [29, 24, 15, 18, 14]]

y_data = ['The course was effectively<br>organized',
          'The course developed my<br>abilities and skills ' +
          'for<br>the subject', 'The course developed ' +
          'my<br>ability to think critically about<br>the subject',
          'I would recommend this<br>course to a friend']

fig = go.Figure()

for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        fig.add_trace(go.Bar(
            x=[xd[i]], y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(color='rgb(248, 248, 249)', width=1)
            )
        ))

fig.update_layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    ),
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    margin=dict(l=120, r=10, t=140, b=80),
    showlegend=False,
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0]) + '%',
                            font=dict(family='Arial', size=14,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd,
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

fig.update_layout(annotations=annotations)

fig.show()


# In[ ]:


y_saving = [1.3586, 2.2623000000000002, 4.9821999999999997, 6.5096999999999996,
            7.4812000000000003, 7.5133000000000001, 15.2148, 17.520499999999998
            ]
y_net_worth = [93453.919999999998, 81666.570000000007, 69889.619999999995,
               78381.529999999999, 141395.29999999999, 92969.020000000004,
               66090.179999999993, 122379.3]
x = ['Japan', 'United Kingdom', 'Canada', 'Netherlands',
     'United States', 'Belgium', 'Sweden', 'Switzerland']


# Creating two subplots
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(go.Bar(
    x=y_saving,
    y=x,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    ),
    name='Household savings, percentage of household disposable income',
    orientation='h',
), 1, 1)

fig.append_trace(go.Scatter(
    x=y_net_worth, y=x,
    mode='lines+markers',
    line_color='rgb(128, 0, 128)',
    name='Household net worth, Million USD/capita',
), 1, 2)

fig.update_layout(
    title='Household savings & net worth for eight OECD countries',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.42],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=25000,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

y_s = np.round(y_saving, decimals=2)
y_nw = np.rint(y_net_worth)

# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, x):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn - 20000,
                            text='{:,}'.format(ydn) + 'M',
                            font=dict(family='Arial', size=12,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 3,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))
# Source
annotations.append(dict(xref='paper', yref='paper',
                        x=-0.2, y=-0.109,
                        text='OECD "' +
                             '(2015), Household savings (indicator), ' +
                             'Household net worth (indicator). doi: ' +
                             '10.1787/cfc6f499-en (Accessed on 05 June 2015)',
                        font=dict(family='Arial', size=10, color='rgb(150,150,150)'),
                        showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()


# ## **Gantt Charts**
# 
# A Gantt chart is a type of a bar chart that illustrates a project schedule. The chart lists the tasks to be performed on the vertical axis, and time intervals on the horizontal axis. The width of the horizontal bars in the graph shows the duration of each activity.
# 
# See also the bar charts examples.

# In[ ]:


import plotly.figure_factory as ff

df = [dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28'),
      dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15'),
      dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30')]

fig = ff.create_gantt(df)
fig.show()


# In[ ]:


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gantt_example.csv')

fig = ff.create_gantt(df, colors=['#333F44', '#93e4c1'], index_col='Complete',
                      show_colorbar=True, bar_width=0.2, showgrid_x=True, showgrid_y=True)
fig.show()


# In[ ]:


df = [
    dict(Task='Morning Sleep', Start='2016-01-01', Finish='2016-01-01 6:00:00', Resource='Sleep'),
    dict(Task='Breakfast', Start='2016-01-01 7:00:00', Finish='2016-01-01 7:30:00', Resource='Food'),
    dict(Task='Work', Start='2016-01-01 9:00:00', Finish='2016-01-01 11:25:00', Resource='Brain'),
    dict(Task='Break', Start='2016-01-01 11:30:00', Finish='2016-01-01 12:00:00', Resource='Rest'),
    dict(Task='Lunch', Start='2016-01-01 12:00:00', Finish='2016-01-01 13:00:00', Resource='Food'),
    dict(Task='Work', Start='2016-01-01 13:00:00', Finish='2016-01-01 17:00:00', Resource='Brain'),
    dict(Task='Exercise', Start='2016-01-01 17:30:00', Finish='2016-01-01 18:30:00', Resource='Cardio'),
    dict(Task='Post Workout Rest', Start='2016-01-01 18:30:00', Finish='2016-01-01 19:00:00', Resource='Rest'),
    dict(Task='Dinner', Start='2016-01-01 19:00:00', Finish='2016-01-01 20:00:00', Resource='Food'),
    dict(Task='Evening Sleep', Start='2016-01-01 21:00:00', Finish='2016-01-01 23:59:00', Resource='Sleep')
]

colors = dict(Cardio = 'rgb(46, 137, 205)',
              Food = 'rgb(114, 44, 121)',
              Sleep = 'rgb(198, 47, 105)',
              Brain = 'rgb(58, 149, 136)',
              Rest = 'rgb(107, 127, 135)')

fig = ff.create_gantt(df, colors=colors, index_col='Resource', title='Daily Schedule',
                      show_colorbar=True, bar_width=0.8, showgrid_x=True, showgrid_y=True)
fig.show()


# ## **Basic Pie Chart**
# 
# A pie chart **<span style="color:limegreen">go.Pie</span>** object is a circular statistical chart, which is divided into sectors to illustrate numerical proportion. Data visualized by the sectors of the pie is set in values. The sector labels are set in labels. The sector colors are set in <span style="color:red">market.colors</span>.

# In[ ]:


labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
values = [4500, 2500, 1053, 500]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()


# In[ ]:


labels = ["Asia", "Europe", "Africa", "Americas", "Oceania"]

fig = make_subplots(1, 2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                    subplot_titles=['1980', '2007'])
fig.add_trace(go.Pie(labels=labels, values=[4, 7, 1, 7, 0.5], scalegroup='one',
                     name="World GDP 1980"), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=[21, 15, 3, 19, 1], scalegroup='one',
                     name="World GDP 2007"), 1, 2)

fig.update_layout(title_text='World GDP')
fig.show()


# ## **Basic Sunburst Plot**
# 
# Sunburst plot visualizes hierarchical data spanning outwards radially from root to leaves. The sunburst sectors are determined by the entries in "labels" and in "parents". The root starts from the center and children are added to the outer rings.
# 
# Main arguments:
# 
# 1. <span style="color:red">labels</span>: sets the labels of sunburst sectors.
# 2. <span style="color:red">parents</span>: sets the parent sectors of sunburst sectors. An empty string **<span style="color:green">''</span>** is used for the root node in the hierarchy. In this example, the root is **<span style="color:green">'Eve'</span>**.
# 3. <span style="color:red">values</span>: sets the values associated with sunburst sectors, determining their width.

# In[ ]:


df1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/sunburst-coffee-flavors-complete.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/coffee-flavors.csv')

fig = go.Figure()

fig.add_trace(go.Sunburst(
    ids=df1.ids,
    labels=df1.labels,
    parents=df1.parents,
    domain=dict(column=0)
))

fig.add_trace(go.Sunburst(
    ids=df2.ids,
    labels=df2.labels,
    parents=df2.parents,
    domain=dict(column=1),
    maxdepth=2
))

fig.update_layout(
    grid= dict(columns=2, rows=1),
    margin = dict(t=0, l=0, r=0, b=0)
)

fig.show()


# ## **Tables**
# 
# **<span style="color:purple">go.Table</span>** provides a **<span style="color:limegreen">Table</span>** object for detailed data viewing. The data is arranged in a grid of rows and columns. Most styling can be specified for header, columns, rows or individual cells. Table is using a column-major order, ie. the grid is represented as a vector of column vectors.
# 
# Note that **<span style="color:violet">Dash</span>** provides a different type of DataTable.

# In[ ]:


fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
                 cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))
                     ])
fig.show()


# In[ ]:


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
               fill_color='lavender',
               align='left'))
])

fig.show()


# ## **Sankey Diagram**
# A Sankey diagram is a flow diagram, in which the width of arrows is proportional to the flow quantity.

# In[ ]:


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = ["A1", "A2", "B1", "B2", "C1", "C2"],
      color = "blue"
    ),
    link = dict(
      source = [0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A2, B1, ...
      target = [2, 3, 3, 4, 4, 5],
      value = [8, 4, 2, 8, 4, 2]
  ))])

fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.show()


# In[ ]:


url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())

fig = go.Figure(data=[go.Sankey(
    valueformat = ".0f",
    valuesuffix = "TWh",
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(color = "black", width = 0.5),
      label =  data['data'][0]['node']['label'],
      color =  data['data'][0]['node']['color']
    ),
    link = dict(
      source =  data['data'][0]['link']['source'],
      target =  data['data'][0]['link']['target'],
      value =  data['data'][0]['link']['value'],
      label =  data['data'][0]['link']['label']
  ))])

fig.update_layout(
    title="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",
    font=dict(size = 10, color = 'white'),
    plot_bgcolor='black',
    paper_bgcolor='black'
)

fig.show()


# # **Exploring ML Data Set**
# We'll work with the popular adult data set. The data set has been taken from UCI Machine Learning Repository. You can download the data from here. In this data set, the dependent variable is <span style="color:orangered">target</span>. It is a binary classification problem. We need to predict if the salary of a given person is less than or more than 50K.

# In[ ]:


# Load the data
train  = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# Check data set
train.info()


# We see that, the train data has <span style="color:blue">32561</span> rows and <span style="color:blue">15</span> columns. Out of these 15 columns, 6 have integers classes and the rest have object (or character) classes. Similarly, we can check for test data. An alternative way of quickly checking rows and columns is

# In[ ]:


print ("The train data has",train.shape)
print ("The test data has",test.shape)
('The train data has', (32561, 15))
('The test data has', (16281, 15))


# Let's have a glimpse of the data set
train.head()


# Now, let's check the missing values (if present) in this data.

# In[ ]:


nans = train.shape[0] - train.dropna().shape[0]
print ("%d rows have missing values in the train data" %nans)

nand = test.shape[0] - test.dropna().shape[0]
print ("%d rows have missing values in the test data" %nand)


# We should be more curious to know which columns have missing values.

# In[ ]:


# Only 3 columns have missing values
train.isnull().sum()


# Let's count the number of unique values from character variables.

# In[ ]:


cat = train.select_dtypes(include=['O'])
cat.apply(pd.Series.nunique)


# Since missing values are found in all 3 character variables, let's impute these missing values with their respective modes.

# In[ ]:


# Education
train.workclass.value_counts(sort=True)
train.workclass.fillna('Private',inplace=True)

# Occupation
train.occupation.value_counts(sort=True)
train.occupation.fillna('Prof-specialty',inplace=True)

# Native Country
train['native.country'].value_counts(sort=True)
train['native.country'].fillna('United-States',inplace=True)


# Let's check again if there are any missing values left.

# In[ ]:


train.isnull().sum()


# Now, we'll check the target variable to investigate if this data is imbalanced or not.

# In[ ]:


# Check proportion of target variable
train.target.value_counts()/train.shape[0]


# We see that 75% of the data set belongs to <=50K class. This means that even if we take a rough guess of target prediction as <=50K, we'll get 75% accuracy. Isn't that amazing? Let's create a cross tab of the target variable with education. With this, we'll try to understand the influence of education on the target variable.

# In[ ]:


pd.crosstab(train.education, train.target,margins=True)/train.shape[0]


# We see that out of 75% people with <=50K salary, 27% people are high school graduates, which is correct as people with lower levels of education are expected to earn less. On the other hand, out of 25% people with >=50K salary, 6% are bachelors and 5% are high-school grads. Now, this pattern seems to be a matter of concern. That's why we'll have to consider more variables before coming to a conclusion.
# 
# If you've come this far, you might be curious to get a taste of building your first machine learning model. In the coming week we'll share an exclusive tutorial on machine learning in python. However, let's get a taste of it here.
# 
# We'll use the famous and formidable scikit learn library. Scikit learn accepts data in numeric format. Now, we'll have to convert the character variable into numeric. We'll use the labelencoder function.
# 
# In label encoding, each unique value of a variable gets assigned a number, i.e., let's say a variable color has four values <span style="color:green">['red','green','blue','pink']</span>.
# 
# Label encoding this variable will return output as: <span style="color:red">red = 2, green = 0, blue = 1, pink = 3.</span>

# In[ ]:


# Load sklearn and encode all object type variables
from sklearn import preprocessing

for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))


# Let's check the changes applied to the data set.

# In[ ]:


train.head()


# As we can see, all the variables have been converted to numeric, including the target variable.

# In[ ]:


# <50K = 0 and >50K = 1
train.target.value_counts()


# # **Building a Random Forest Model**
# Let's create a random forest model and check the model's accuracy.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

y = train['target']
del train['target']

X = train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

#train the RF classifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)
clf.fit(X_train,y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

clf.predict(X_test)


# Now, let's make prediction on the test set and check the model's accuracy.

# In[ ]:


# Make prediction and check model's accuracy
prediction = clf.predict(X_test)
acc =  accuracy_score(np.array(y_test),prediction)
print ('The accuracy of Random Forest is {}'.format(acc))


# Hurrah! Our learning algorithm gave 85% accuracy. Well, we can do tons of things on this data and improve the accuracy. We'll learn about it in future articles. What's next?
# 
# In this tutorial, we divided the train data into two halves and made prediction on the test data. As your exercise, you should use this model and make prediction on the test data we loaded initially. You can perform same set of steps we did on the train data to complete this exercise.

# # **Summary**
# 
# This tutorial is meant to help **<span style="color:turquoise">Python</span>** developers or anyone who's starting with **<span style="color:turquoise">Python</span>** to get a taste of data manipulation and a little bit of machine learning using **<span style="color:turquoise">Python</span>**. I'm sure, by now you would be convinced that **<span style="color:turquoise">Python</span>** is actually very powerful in handling and processing data sets. But, what we learned here is just the tip of the iceberg. Don't get complacent with this knowledge.

# **<span style="color:magenta">Text Mining:</span>** https://towardsdatascience.com/gentle-start-to-natural-language-processing-using-python-6e46c07addf3
# 
# **<span style="color:magenta">Automated Python Scripts in Windows:</span>** https://datatofish.com/python-script-windows-scheduler/
