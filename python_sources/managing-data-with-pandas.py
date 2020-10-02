#!/usr/bin/env python
# coding: utf-8

# # The Pandas Library

# https://dhafermalouche.net

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Description-from-the-Pandas-documentation:" data-toc-modified-id="Description-from-the-Pandas-documentation:-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Description from the Pandas documentation:</a></span></li><li><span><a href="#Series-and-DataFrames" data-toc-modified-id="Series-and-DataFrames-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Series and DataFrames</a></span><ul class="toc-item"><li><span><a href="#The-Panda-Series" data-toc-modified-id="The-Panda-Series-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>The Panda Series</a></span></li><li><span><a href="#Pandas-DataFrames" data-toc-modified-id="Pandas-DataFrames-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Pandas DataFrames</a></span></li></ul></li><li><span><a href="#Data-Manipulation" data-toc-modified-id="Data-Manipulation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Manipulation</a></span></li></ul></div>

# ## Description from the Pandas documentation:
# pandas is a data analysis library providing fast, flexible, and expressive data structures designed to work with relational or table-like data (SQL table or Excel spreadsheet). It is a fundamental high-level building block for doing practical, real world data analysis in Python.
# pandas is well suited for:
# Tabular data with heterogeneously-typed columns, as in an SQL table or Excel spreadsheet
# Ordered and unordered (not necessarily fixed-frequency) time series data.
# Arbitrary matrix data (homogeneously typed or heterogeneous) with row and column labels
# Any other form of observational / statistical data sets. The data actually need not be labeled at all to be placed into a pandas data structure
# The two primary data structures of pandas, Series (1-dimensional) and DataFrame (2-dimensional), handle the vast majority of typical use cases in finance, statistics, social science, and many areas of engineering. Pandas is built on top of NumPy and is intended to integrate well within a scientific computing environment with many other 3rd party libraries.

# Here are just a few of the things that pandas does well:

# + Easy handling of missing data (represented as NaN) in floating point as well as non-floating point data
# + Size mutability: columns can be inserted and deleted from DataFrame and higher dimensional objects
# + Automatic and explicit data alignment: objects can be explicitly aligned to a set of labels, or the user can simply ignore the labels and let Series, DataFrame, etc. automatically align the data for you in computations
# + Powerful, flexible group by functionality to perform split-apply-combine operations on data sets, for both aggregating and transforming data
# + Make it easy to convert ragged, differently-indexed data in other Python and NumPy data structures into DataFrame objects
# + Intelligent label-based slicing, fancy indexing, and subsetting of large data sets
# + Intuitive merging and joining data sets
# + Flexible reshaping and pivoting of data sets
# + Hierarchical labeling of axes (possible to have multiple labels per tick)
# + Robust IO tools for loading data from flat files (CSV and delimited), Excel files, databases, and saving / loading data from the ultrafast [HDF5 format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)
# + Time series-specific functionality: date range generation and frequency conversion, moving window statistics, moving window linear regressions, date shifting and lagging, etc.

# ## Series and DataFrames

# In[ ]:


import pandas as pd


# ### The Panda Series 

# The Series data structure in Pandas is a one-dimensional labeled array.
# + Data in the array can be of any type (integers, strings, floating point numbers, Python objects, etc.).
# + Data within the array is homogeneous
# + Pandas Series objects always have an index: this gives them both ndarray-like and dict-like properties.

# Creating a Panda Serie:
# + Creation from a list
# + Creation from a dictionary
# + Creation from a ndarray
# + From an external source like a file

# **From a list**

# In[ ]:


temperature = [34, 56, 15, -9, -121, -5, 39]
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# create series 
series_from_list = pd.Series(temperature, index=days)
series_from_list


# The series should contains homogeneous types

# In[ ]:


temperature = [34, 56, 'a', -9, -121, -5, 39]
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


# We create series 
# 

# In[ ]:


series_from_list = pd.Series(temperature, index=days)
series_from_list


# **from a dictionary**

# In[ ]:


my_dict = {'Mon': 33, 'Tue': 19, 'Wed': 15, 'Thu': 89, 'Fri': 11, 'Sat': -5, 'Sun': 9}
my_dict


# In[ ]:


series_from_dict = pd.Series(my_dict)
series_from_dict


# **From a numpy array**

# In[ ]:


import numpy as np


# In[ ]:


my_array = np.linspace(0,10,15)
my_array


# In[ ]:


series_from_ndarray = pd.Series(my_array)
series_from_ndarray


# ### Pandas DataFrames 

# DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dict of Series objects.
# You can create a DataFrame from:
# + Dict of 1D ndarrays, lists, dicts, or Series
# + 2-D numpy.ndarray
# + From text, CSV, Excel files or databases
# + Many other ways
# 
# Reading the data. 
# 
# Sample data: HR Employee Attrition and Performance You can get it from here and add it to your working directory:
# 
# https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/

# Importing the xlsx file by considering the variable EmployeeNumber as an Index variable

# In[ ]:


data = pd.read_excel(io="../input/WA_Fn-UseC_-HR-Employee-Attrition.xlsx", sheetname=0, index_col='EmployeeNumber')


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data['Attrition'].head()


# ## Data Manipulation

# In[ ]:


data[['Age', 'Gender','YearsAtCompany']].head()


# In[ ]:


data['AgeInMonths'] = 12*data['Age']
data['AgeInMonths'].head()


# In[ ]:


del data['AgeInMonths']


# In[ ]:


data.columns


# In[ ]:


data['BusinessTravel'][10:15]


# In[ ]:


data[10:15]


# In[ ]:


selected_EmployeeNumbers = [15, 94, 337, 1120]


# In[ ]:


data['YearsAtCompany'].loc[selected_EmployeeNumbers]


# In[ ]:


data.loc[selected_EmployeeNumbers]


# In[ ]:


data.loc[94,'YearsAtCompany']


# In[ ]:


data['Department'].value_counts()


# In[ ]:


data['Department'].value_counts().plot(kind='barh', title='Department')


# In[ ]:


data['Department'].value_counts().plot(kind='pie', title='Department')


# In[ ]:


data['Attrition'].value_counts()


# In[ ]:


data['Attrition'].value_counts(normalize=True)


# In[ ]:


data['HourlyRate'].mean()


# What's the overall statisfaction of the Employees?

# In[ ]:


data['JobSatisfaction'].head()


# Let us change the levels of the variable satisfaction

# In[ ]:


JobSatisfaction_cat = {
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very High'
}


# In[ ]:


data['JobSatisfaction'] = data['JobSatisfaction'].map(JobSatisfaction_cat)
data['JobSatisfaction'].head()


# In[ ]:


data['JobSatisfaction'].value_counts()


# In[ ]:


100*data['JobSatisfaction'].value_counts(normalize=True)


# In[ ]:


data['JobSatisfaction'].value_counts(normalize=True).plot(kind='pie', title='Department')


# In[ ]:


data['JobSatisfaction'] = data['JobSatisfaction'].astype(dtype='category', 
                               categories=['Low', 'Medium', 'High', 'Very High'],
                               ordered=True)


# In[ ]:


data['JobSatisfaction'].head()


# In[ ]:


data['JobSatisfaction'].value_counts().plot(kind='barh', title='Department')


# In[ ]:


data['JobSatisfaction'].value_counts(sort=False).plot(kind='barh', title='Department')


# In[ ]:


data['JobSatisfaction'] == 'Low'


# In[ ]:


data.loc[data['JobSatisfaction'] == 'Low'].index


# In[ ]:


data['JobInvolvement'].head()


# In[ ]:


subset_of_interest = data.loc[(data['JobSatisfaction'] == "Low") | (data['JobSatisfaction'] == "Very High")]
subset_of_interest.shape


# In[ ]:


subset_of_interest['JobSatisfaction'].value_counts()


# Let's then remove the categories or levels that we won't use

# In[ ]:


subset_of_interest['JobSatisfaction'].cat.remove_unused_categories(inplace=True)


# In[ ]:


grouped = subset_of_interest.groupby('JobSatisfaction')


# In[ ]:


grouped.groups


# The Low statisfaction group

# In[ ]:


grouped.get_group('Low').head()


# and the Very High satisfaction group

# In[ ]:


grouped.get_group('Very High').head()


# **The average of the Age of each group**

# In[ ]:


grouped['Age']


# In[ ]:


grouped['Age'].mean()


# In[ ]:


grouped['Age'].describe()


# In[ ]:


grouped['Age'].describe().unstack()


# **Comparing densities**

# In[ ]:


grouped['Age'].plot(kind='density', title='Age')


# **By Department**

# In[ ]:


grouped['Department'].value_counts().unstack()


# We can normalize it

# In[ ]:


grouped['Department'].value_counts(normalize=True).unstack()


# In[ ]:


grouped['Department'].value_counts().unstack().plot(kind="barh")


# In[ ]:


grouped['Department'].value_counts(normalize=True).unstack().plot(kind="barh")


# We can compare it with the whole sample

# In[ ]:


data['Department'].value_counts(normalize=True,sort=False).plot(kind="barh")


# But the colors and the order don't match with the other bar chart. We need to reorder the Department variable

# In[ ]:


data['Department'] = data['Department'].astype(dtype='category', 
                               categories=['Human Resources', 'Research & Development', 'Sales'],
                               ordered=True)


# In[ ]:


data['Department'].value_counts(normalize=True,sort=False).plot(kind="barh")


# In[ ]:


grouped['DistanceFromHome'].describe().unstack()


# In[ ]:


grouped['DistanceFromHome'].plot(kind='density', title='Distance From Home',legend=True)


# In[ ]:


grouped['HourlyRate'].describe()


# In[ ]:


grouped['HourlyRate'].plot(kind='density', title='Hourly Rate',legend=True)


# In[ ]:


grouped['MonthlyIncome'].describe()


# In[ ]:


grouped['HourlyRate'].plot(kind='density', title='Hourly Rate',legend=True)

