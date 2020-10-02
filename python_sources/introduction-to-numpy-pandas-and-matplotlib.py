#!/usr/bin/env python
# coding: utf-8

# # <span style="color:red"> Introduction to NumPy, Pandas and Matplotlib </span>

# ## <span style="color:blue"> Data Analysis </span>

# Data Analysis is a process of inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information, suggesting conclusions, and supporting decision-making.

# Stpes for Data Analysis, Data Manipulation and Data Visualization:
# 1. Tranform Raw Data in a Desired Format 
# 2. Clean the Transformed Data (Step 1 and 2 also called as a Pre-processing of Data)
# 3. Prepare a Model
# 4. Analyse Trends and Make Decisions

# ## <span style="color:blue"> NumPy </span>

# NumPy is a package for scientific computing.
# 1. Multi dimensional array
# 2. Methods for processing arrays
# 3. Element by element operations
# 4. Mathematical operations like logical, Fourier transform, shape manipulation, linear algebra and random number generation

# In[ ]:


import numpy as np


# ### <span style="color:orange"> Ndarray - NumPy Array </span>

# The ndarray is a multi-dimensional array object consisting of two parts -- the actual data, some metadata which describes the stored data. They are indexed just like sequence are in Python, starting from 0
# 1. Each element in ndarray is an object of data-type object called dtype
# 2. An item extracted from ndarray, is represented by a Python object of an array scalar type

# ### <span style="color:green"> Single Dimensional Array </span>

# ### <span style="color:orange"> Creating a Numpy Array </span>

# In[ ]:


# Creating a single-dimensional array
a = np.array([1,2,3]) # Calling the array function
print(a)


# In[ ]:


# Creating a multi-dimensional array
# Each set of elements within a square bracket indicates a row
# Array of two rows and two columns
b = np.array([[1,2], [3,4]])
print(b)


# In[ ]:


# Creating an ndarray by wrapping a list
list1 = [1,2,3,4,5] # Creating a list
arr = np.array(list1) # Wrapping the list
print(arr)


# In[ ]:


# Creating an array of numbers of a specified range
arr1 = np.arange(10, 100) # Array of numbers from 10 up to and excluding 100
print(arr1)


# In[ ]:


# Creating a 5x5 array of zeroes
arr2 = np.zeros((5,5))
print(arr2)


# In[ ]:


# Creating a linearly spaced vector, with spacing
vector = np.linspace(0, 20, 5) # Start, stop, step
print(vector)


# In[ ]:


# Creating Arrays from Existing Data
x = [1,2,3]
# Used for converting Python sequences into ndarrays
c = np.asarray(x) #np.asarray(a, dtype = None, order = None)
print(c)


# In[ ]:


# Converting a linear array of 8 elements into a 2x2x2 3D array
arr3 = np.zeros(8) # Flat array of eight zeroes
arr3d = arr3.reshape((2,2,2)) # Restructured array
print(arr3d)


# In[ ]:


# Flatten rgw 3d array to get back the linear array
arr4 = arr3d.ravel()
print(arr4)


# ### <span style="color:orange"> Indexing of NumPy Arrays </span>

# In[ ]:


# NumPy array indexing is identical to Python's indexing scheme
arr5 = np.arange(2, 20)
element = arr5[6]
print(element)


# In[ ]:


# Python's concept of lists slicing is extended to NumPy.
# The slice object is constructed by providing start, stop, and step parameters to slice()
arr6 = np.arange(20)
arr_slice = slice(1, 10, 2) # Start, stop & step
element2 = arr6[6]
print(arr6[arr_slice])


# In[ ]:


# Slicing items beginning with a specified index
arr7 = np.arange(20)
print(arr7[2:])


# In[ ]:


# Slicing items until a specified index
print(arr7[:15])


# In[ ]:


# Extracting specific rows and columns using Slicing
d = np.array([[1,2,3], [3,4,5], [4,5,6]])
print(d[0:2, 0:2]) # Slice the first two rows and the first two columns


# ### <span style="color:orange"> NumPy Array Attributes </span>

# In[ ]:


print(d.shape) # Returns a tuple consisting of array dimensions
print(d.ndim) # Attribute returns the number of array dimensions
print(a.itemsize) # Returns the length of each element of array in bytes


# In[ ]:


y = np.empty([3,2], dtype = int) # Creates an uninitialized array of specified shape and dtype
print(y)


# In[ ]:


# Returns a new array of specified size, filled with zeros
z = np.zeros(5) # np.zeros(shape, dtype = float)
print(z)


# ### <span style="color:orange"> Reading & Writing from Files </span>

# In[ ]:


# NumPy provides the option of importing data from files directly into ndarray using the loadtxt function
# The savetxt function can be used to write data from an array into a text file
#import os
#print(os.listdir('../input'))
arr_txt = np.loadtxt('../input/data_file.txt')
np.savetxt('newfilex.txt', arr_txt)


# In[ ]:


# NumPy arrays can be dumped into CSV files using the savetxt function and the comma delimiter
# The genfromtxt function can be used to read data from a CSV file into a NumPy array
arr_csv = np.genfromtxt('../input/Hurricanes.csv', delimiter = ',')
np.savetxt('newfilex.csv', arr_csv, delimiter = ',')


# ## <span style="color:blue"> Pandas </span>

# Pandas is an open-source Python library providing efficient, easy-to-use data structure and data analysis tools. The name Pandas is derived from "Panel Data" - an Econometrics from Multidimensional Data. Pandas is well suited for many different kinds of data:
# 1. Tabular data with heterogeneously-type columns.
# 2. Ordered and unordered time series data.
# 3. Arbitary matrix data with row and column labels.
# 4. Any other form observational/statistical data sets. The data actually need not be labeled at all to be placed into a pandas data structure.

# Pandas provides three data structure - all of which are build on top of the NumPy array - all the data structures are value-mutable
# 1. Series (1D) - labeled, homogenous array of immutable size
# 2. DataFrames (2D) - labeled, heterogeneously typed, size-mutable tabular data structures
# 3. Panels (3D) - Labeled, size-mutable array

# In[ ]:


import pandas as pd


# ### <span style="color:orange"> Series </span>

# 1. A Series is a single-dimensional array structures that stores homogenous data i.e., data of a single type.
# 2. All the elements of a Series are value-mutable and size-immutable
# 3. Data can be of multiple data types such as ndarray, lists, constants, series, dict etc.
# 4. Indexes must be unique, hashable and have the same length as data. Defaults to np.arrange(n) if no index is passed.
# 5. Data type of each column; if none is mentioned, it will be inferred; automatically
# 6. Deep copies data, set to false as default

# ### <span style="color:orange"> Creating a Series </span>

# In[ ]:


# Creating an empty Series
series = pd.Series() # The Series() function creates a new Series
print(series)


# In[ ]:


# Creating a series from an ndarray
# Note that indexes are a assigned automatically if not specifies
arr = np.array([10,20,30,40,50])
series1 = pd.Series(arr)
print(series1)


# In[ ]:


# Creating a series from a Python dict
# Note that the keys of the dictionary are used to assign indexes during conversion
data = {'a':10, 'b':20, 'c':30}
series2 = pd.Series(data)
print(series2)


# In[ ]:


# Retrieving a part of the series using slicing
print(series1[1:4])


# ### <span style="color:orange"> DataFrames </span>

# 1. A DataFrame is a 2D data structure in which data is aligned in a tabular fashion consisting of rows & columns
# 2. A DataFrame can be created using the following constructor - pandas.DataFrame(data, index, dtype, copy)
# 3. Data can be of multiple data types such as ndarray, list, constants, series, dict etc.
# 4. Index Row and column labels of the dataframe; defaults to np.arrange(n) if no index is passed
# 5. Data type of each column
# 6. Creates a deep copy of the data, set to false as default

# ### <span style="color:orange"> Creating a DataFrame </span>

# In[ ]:


# Converting a list into a DataFrame
list1 = [10, 20, 30, 40]
table = pd.DataFrame(list1)
print(table)


# In[ ]:


# Creating a DataFrame from a list of dictionaries
data = [{'a':1, 'b':2}, {'a':2, 'b':4, 'c':8}]
table1 = pd.DataFrame(data)
print(table1)
# NaN (not a number) is stored in areas where no data is provided


# In[ ]:


# Creating a DataFrame from a list of dictionaries and accompaying row indices
table2 = pd.DataFrame(data, index = ['first', 'second'])
# Dict keys become column lables
print(table2)


# In[ ]:


# Converting a dictionary of series into a DataFrame
data1 = {'one':pd.Series([1,2,3], index = ['a', 'b', 'c']),
        'two':pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'])}
table3 = pd.DataFrame(data1)
print(table3)
# the resultant index is the union of all the series indexes passed


# ### <span style="color:orange"> DataFrame - Addition & Deletion of Columns </span>

# In[ ]:


# A new column can be added to a DataFrame when the data is passed as a Series
table3['three'] = pd.Series([10,20,30], index = ['a', 'b', 'c'])
print(table3)


# In[ ]:


# DataFrame columns can be deleted using the del() function
del table3['one']
print(table3)


# In[ ]:


# DataFrame columns can be deleted using the pop() function
table3.pop('two')
print(table3)


# ### <span style="color:orange"> DataFrame - Addition & Deletion of Rows </span>

# In[ ]:


# DataFrame rows can be selected by passing the row lable to the loc() function
print(table3.loc['c'])


# In[ ]:


# Row selection can also be done using the row index
print(table3.iloc[2])


# In[ ]:


# The append() function can be used to add more rows to the DataFrame
data2 = {'one':pd.Series([1,2,3], index = ['a', 'b', 'c']),
        'two':pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'])}
table5 = pd.DataFrame(data2)
table5['three'] = pd.Series([10,20,30], index = ['a', 'b', 'c'])
row = pd.DataFrame([[11,13],[17,19]], columns = ['two', 'three'])
table6 = table5.append(row)
print(table6)


# In[ ]:


# The drop() function can be used to drop rows whose labels are provided
table7 = table6.drop('a')
print(table7)


# ### <span style="color:orange"> Importing & Exporting Data </span>

# In[ ]:


# Data can be loaded into DataFrames from input data stored in the CSV format using the read_csv() function
table_csv = pd.read_csv('../input/Cars2015.csv')


# In[ ]:


# Data present in DataFrames can be written to a CSV file using the to_csv() function
# If the specified path doesn't exist, a file of the same name is automatically created
table_csv.to_csv('newcars2015.csv')


# In[ ]:


# Data can be loaded into DataFrames from input data stored in the Excelsheet format using read_excel()
sheet = pd.read_excel('cars2015.xlsx')


# In[ ]:


# Data present in DataFrames can be written to a spreadsheet file using to_excel()
#If the specified path doesn't exist, a file of the same name is automatically created
sheet.to_excel('newcars2015.xlsx')


# ## <span style="color:blue"> Matplotlib </span>

# 1. Matplotlib is a Python library that is specially designed for the development of graphs, charts etc., in order to provide interactive data visualisation
# 2. Matplotlib is inspired from the MATLAB software and reproduces many of it's features

# In[ ]:


# Import Matplotlib submodule for plotting
import matplotlib.pyplot as plt


# ### <span style="color:orange"> Plotting in Matplotlib </span>

# In[ ]:


plt.plot([1,2,3,4]) # List of vertical co-ordinates of the points plotted
plt.show() # Displays plot
# Implicit X-axis values from 0 to (N-1) where N is the length of the list


# In[ ]:


# We can specify the values for both axes
x = range(5) # Sequence of values for the x-axis
# X-axis values specified - [0,1,2,3,4]
plt.plot(x, [x1**2 for x1 in x]) # vertical co-ordinates of the points plotted: y = x^2
plt.show()


# In[ ]:


# We can use NumPy to specify the values for both axes with greater precision
x = np.arange(0, 5, 0.01)
plt.plot(x, [x1**2 for x1 in x]) # vertical co-ordinates of the points plotted: y = x^2
plt.show()


# ### <span style="color:orange"> Multiline Plots </span>

# In[ ]:


# Multiple functions can be drawn on the same plot
x = range(5)
plt.plot(x, [x1 for x1 in x])
plt.plot(x, [x1*x1 for x1 in x])
plt.plot(x, [x1*x1*x1 for x1 in x])
plt.show()


# In[ ]:


# Different colours are used for different lines
x = range(5)
plt.plot(x, [x1 for x1 in x],
         x, [x1*x1 for x1 in x],
         x, [x1*x1*x1 for x1 in x])
plt.show()


# ### <span style="color:orange"> Grids </span>

# In[ ]:


# The grid() function adds a grid to the plot
# grid() takes a single Boolean parameter
# grid appears in the background of the plot
x = range(5)
plt.plot(x, [x1 for x1 in x],
         x, [x1*2 for x1 in x],
         x, [x1*4 for x1 in x])
plt.grid(True)
plt.show()


# ### <span style="color:orange"> Limiting the Axes </span>

# In[ ]:


# The scale of the plot can be set using axis()
x = range(5)
plt.plot(x, [x1 for x1 in x],
         x, [x1*2 for x1 in x],
         x, [x1*4 for x1 in x])
plt.grid(True)
plt.axis([-1, 5, -1, 10]) # Sets new axes limits
plt.show()


# In[ ]:


# The scale of the plot can also be set using xlim() and ylim()
x = range(5)
plt.plot(x, [x1 for x1 in x],
         x, [x1*2 for x1 in x],
         x, [x1*4 for x1 in x])
plt.grid(True)
plt.xlim(-1, 5)
plt.ylim(-1, 10)
plt.show()


# ### <span style="color:orange"> Adding Labels </span>

# In[ ]:


# Labels can be added to the axes of the plot
x = range(5)
plt.plot(x, [x1 for x1 in x],
         x, [x1*2 for x1 in x],
         x, [x1*4 for x1 in x])
plt.grid(True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# ### <span style="color:orange"> Adding the Title </span>

# In[ ]:


# The title defines the data plotted on the graph
x = range(5)
plt.plot(x, [x1 for x1 in x],
         x, [x1*2 for x1 in x],
         x, [x1*4 for x1 in x])
plt.grid(True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("Polynomial Graph") # Pass the title as a parameter to title()
plt.show()


# ### <span style="color:orange"> Adding a Legend </span>

# In[ ]:


# Legends explain the meaning of each line in the graph
x = np.arange(5)
plt.plot(x, x, label = 'linear')
plt.plot(x, x*x, label = 'square')
plt.plot(x, x*x*x, label = 'cube')
plt.grid(True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("Polynomial Graph")
plt.legend()
plt.show()


# ### <span style="color:orange"> Adding a Markers </span>

# In[ ]:


x = [1, 2, 3, 4, 5, 6]
y = [11, 22, 33, 44, 55, 66]
plt.plot(x, y, 'bo')
for i in range(len(x)):
    x_cord = x[i]
    y_cord = y[i]
    plt.text(x_cord, y_cord, (x_cord, y_cord), fontsize = 10)
plt.show()


# ### <span style="color:orange"> Saving Plots </span>

# In[ ]:


# Plots can be saved using savefig()
x = np.arange(5)
plt.plot(x, x, label = 'linear')
plt.plot(x, x*x, label = 'square')
plt.plot(x, x*x*x, label = 'cube')
plt.grid(True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("Polynomial Graph")
plt.legend()
plt.savefig('plot.png') # Saves an image names 'plot.png' in the current directory
plt.show()


# ### <span style="color:orange"> Plot Types </span>

# Matplotlib provides many types of plot formats for visualising information
# 1. Scatter Plot
# 2. Histogram
# 3. Bar Graph
# 4. Pie Chart

# ### <span style="color:orange"> Histogram </span>

# In[ ]:


# Histograms display the distribution of a variable over a range of frequencies or values
y = np.random.randn(100, 100) # 100x100 array of a Gaussian distribution
plt.hist(y) # Function to plot the histogram takes the dataset as the parameter
plt.show()


# In[ ]:


# Histogram groups values into non-overlapping categories called bins
# Default bin value of the histogram plot is 10
y = np.random.randn(1000)
plt.hist(y, 100)
plt.show()


# ### <span style="color:orange"> Bar Chart </span>

# In[ ]:


# Bar charts are used to visually compare two or more values using rectangular bars
# Default width of each bar is 0.8 units
# [1,2,3] Mid-point of the lower face of every bar
# [1,4,9] Heights of the successive bars in the plot
plt.bar([1,2,3], [1,4,9])
plt.show()


# In[ ]:


dictionary = {'A':25, 'B':70, 'C':55, 'D':90}
for i, key in enumerate(dictionary):
    plt.bar(i, dictionary[key]) # Each key-value pair is plotted individually as dictionaries are not iterable
plt.show()


# In[ ]:


dictionary = {'A':25, 'B':70, 'C':55, 'D':90}
for i, key in enumerate(dictionary):
    plt.bar(i, dictionary[key])
plt.xticks(np.arange(len(dictionary)), dictionary.keys()) # Adds the keys as labels on the x-axis
plt.show()


# ### <span style="color:orange"> Pie Chart </span>

# In[ ]:


plt.figure(figsize = (3,3)) # Size of the plot in inches
x = [40, 20, 5] # Proportions of the sectors
labels = ['Bikes', 'Cars', 'Buses']
plt.pie(x, labels = labels)
plt.show()


# ### <span style="color:orange"> Scatter Plot </span>

# In[ ]:


# Scatter plots display values for two sets of data, visualised as a collection of points
# Two Gaussion distribution plotted
x = np.random.rand(1000)
y = np.random.rand(1000)
plt.scatter(x, y)
plt.show()


# ### <span style="color:orange"> Styling </span>

# In[ ]:


# Matplotlib allows to choose custom colours for plots
y = np.arange(1, 3)
plt.plot(y, 'y') # Specifying line colours
plt.plot(y+5, 'm')
plt.plot(y+10, 'c')
plt.show()


# Color code:
# 1. b = Blue
# 2. c = Cyan
# 3. g = Green
# 4. k = Black
# 5. m = Magenta
# 6. r = Red
# 7. w = White
# 8. y = Yellow

# In[ ]:


# Matplotlib allows different line styles for plots
y = np.arange(1, 100)
plt.plot(y, '--', y*5, '-.', y*10, ':')
plt.show()
# - Solid line
# -- Dashed line
# -. Dash-Dot line
# : Dotted Line


# In[ ]:


# Matplotlib provides customization options for markers
y = np.arange(1, 3, 0.2)
plt.plot(y, '*',
        y+0.5, 'o',
        y+1, 'D',
        y+2, '^',
        y+3, 's') # Specifying line styling
plt.show()

