#!/usr/bin/env python
# coding: utf-8

# **Coursebook: Exploratory Data Analysis**
# - Part 2 of Data Analytics Specialization
# - Course Length: 12 hours
# - Last Updated: February 2020
# 
# ___
# 
# - Author: [Samuel Chan](https://github.com/onlyphantom)
# - Developed by [Algoritma](https://algorit.ma)'s product division and instructors team

# # Background
# 
# ## Top-Down Approach 
# 
# The coursebook is part of the **Data Analytics Specialization** offered by [Algoritma](https://algorit.ma). It takes a more accessible approach compared to Algoritma's core educational products, by getting participants to overcome the "how" barrier first, rather than a detailed breakdown of the "why". 
# 
# This translates to an overall easier learning curve, one where the reader is prompted to write short snippets of code in frequent intervals, before being offered an explanation on the underlying theoretical frameworks. Instead of mastering the syntactic design of the Python programming language, then moving into data structures, and then the `pandas` library, and then the mathematical details in an imputation algorithm, and its code implementation; we would do the opposite: Implement the imputation, then a succinct explanation of why it works and applicational considerations (what to look out for, what are assumptions it made, when _not_ to use it etc).
# 
# ## Training Objectives
# 
# This coursebook is intended for participants who have completed the preceding courses offered in the **Data Analytics Developer** Specialization. This is the second course, **Exploratory Data Analysis**
# 
# The coursebook focuses on:
# - Date Time objects
# - Categorical data types
# - Why and What: Exploratory Data Analysis
# - Cross Tabulation and Pivot Table
# - Treating Duplicates and Missing Values 
# 
# At the end of this course is a Learn-by-Building section, where you are expected to apply all that you've learned on a new dataset, and attempt the given questions.

# # Data Preparation and Exploration
# 
# About 60 years ago, John Tukey defined data analysis as the "procedures for analyzing data, techniques for interpreting the results of such procedures ... and all the machinery of mathematical statistics which apply to analyzing dsta". His championing of EDA encouraged the development of statsitical computing packages, especially S at Bell Labs (which later inspired R).
# 
# He wrote a book titled _Exploratory Data Analysis_ arguing that too much emphasis in statistics was placed on hypothesis testing (confirmatory data analysis) while not enough was placed on the discovery of the unexpected. 
# 
# > Exploratory data analysis isolates patterns and features of the data and reveals these forcefully to the analyst.
# 
# This course aims to present a selection of EDA techniques -- some developed by John Tukey himself -- but with a special emphasis on its application to modern business analytics.
# 
# In the previous course, we've got our hands on a few common techniques:
# 
# - `.head()` and `.tail()`
# - `.describe()`
# - `.shape` and `.size`
# - `.axes`
# - `.dtypes`
# 
# In the following chapters, we'll expand our EDA toolset with the following additions:  
# 
# - Tables
# - Cross-Tables and Aggregates
# - Using `aggfunc` for aggregate functions
# - Pivot Tables

# In[ ]:


import pandas as pd
import numpy as np
print(pd.__version__)


# ## Working with Datetime
# 
# Given the program's special emphasis on business-driven analytics, one data type of particular interest to us is the `datetime`. In the first part of this coursebook, we've seen an example of `datetime` in the section introducing data types (`employees.joined`).
# 
# A large portion of data science work performed by business executives involve time series and/or dates (think about the kind of data science work done by computer vision researchers, and compare that to the work done by credit rating analysts or marketing executives and this special relationship between business and datetime data becomes apparent), so adding a level of familiarity with this format will serve you well in the long run. 
# 
# As a start, let's read our data,`household.csv`:

# In[ ]:


household = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/household.csv")
household.head()


# In[ ]:


household.dtypes


# Notice that all columns are in the right data types, except for `purchase_time`. The correct data type for this column would have to be a `datetime`.
# 
# To convert a column `x` to a datetime, we would use:
# 
#     `x = pd.to_datetime(x)`
#     

# In[ ]:


household['purchase_time'] = pd.to_datetime(household['purchase_time'])
household.head()


# As you can see from the code above,`pd.to_datetime()` could do the conversion to datetime in a smart way without datetime format string required. Convenient for sure, but for some situation, this manner of `pandas` can be a little tricky.
# 
# Suppose we have a column which stores a daily sales data from end of January to the beginning of February:

# In[ ]:


date = pd.Series(['30-01-2020', '31-01-2020', '01-02-2020','02-02-2020'])
date


# The legal and cultural expectations for datetime format may vary between countries. In Indonesia for example, most people are used to storing dates in DMY order. Why it matters? Let's see what happen next when we convert our `date` to datetime object:

# In[ ]:


pd.to_datetime(date)


# Take a look on the third observation; `pd.to_datetime` converts it to 2nd January while the actual data represents February 2nd. The function may find the string pattern automatically and smartly, but note that for dates with multiple representations, it will infer it as a month first order by default.
# 
# That's why it's important to know that `pd.to_datetime` accepts other parameters, `format` and `dayfirst`:

# In[ ]:


# Solution 1
pd.to_datetime(date, format="%d-%m-%Y")


# Solution 2
pd.to_datetime(date, dayfirst=True)


# Using Python's `datetime` module, `pandas` pass the date string to `.strptime()` and follows by what's called Python's strptime directives. The full list of directives can be found in this [Documentation](https://strftime.org/).

# Other than `to_datetime`, `pandas` has a number of machineries to work with `datetime` objects. These are convenient for when we need to extract the `month`, or `year`, or `weekday_name` from `datetime`. Some common applications in business analysis include:
# 
# - `household['purchase_time'].dt.month`
# - `household['purchase_time'].dt.year`
# - `household['purchase_time'].dt.day`
# - `household['purchase_time'].dt.dayofweek`
# - `household['purchase_time'].dt.hour`
# - `household['purchase_time'].dt.weekday_name` or `household['purchase_time'].dt.day_name()` for pandas v.1.x.x

# **Knowledge Check:** Date time types  
# _Est. Time required: 20 minutes_
# 
# 1. In the following cell, start again by reading in the `household.csv` dataset 
# 2. Convert `purchase_time` to `datetime`. Use `pd.to_datetime()` for this.
# 3. Use `x.dt.weekday_name`, assuming `x` is a datetime object to get the day of week. Assign this to a new column in your `household` Data Frame, name it `weekday`
# 4. Print the first 5 rows of your data to verify that your preprocessing steps are correct

# In[ ]:


## Your code below

## -- Solution code


# Tips: In the cell above, start from:
# 
# `household = pd.read_csv("data_input/household.csv")`
# 
# Inspect the first 5 rows of your data and pay close attention to the `weekday` column. 
# 
# Bonus challenge: How many transactions happen on each day of the week? Use `pd.crosstab(index=__, columns="count")` or `x.value_counts()`.

# In[ ]:


## Your code below


## -- Solution code


# There are also other functions that can be helpful in certain situations. Supposed we want to transform the existing `datetime` column into values of periods we can use the `.to_period` method:
# 
# - `household['purchase_time'].dt.to_period('D')`
# - `household['purchase_time'].dt.to_period('W')`
# - `household['purchase_time'].dt.to_period('M')`
# - `household['purchase_time'].dt.to_period('Q')`

# If you've managed the above exercises, well done! Run the following cell anyway to make sure we're at the same starting point as we go into the next chapter of working with categorical data (factors). 

# In[ ]:


# Reference answer for Knowledge Check
household = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/household.csv", index_col=1, parse_dates=['purchase_time'])
household.drop(['receipt_id', 'yearmonth', 'sub_category'], axis=1, inplace=True)
household['weekday'] = household['purchase_time'].dt.weekday_name
pd.crosstab(index=household['weekday'], columns='count')


# ## Working with Categories

# From the output of `dtypes`, we see that there are three variables currently stored as `object` type where a `category` is more appropriate. This is a common diagnostic step, and one that you will employ in almost every data analysis project.

# In[ ]:


household.dtypes


# We'll convert the `weekday` column to a categorical type using `.astype()`. `astype('int64')` converts a Series to an integer type, and `.astype(category)` logically, converts a Series to a categorical.
# 
# By default, `.astype()` will raise an error if the conversion is not successful (we call them "exceptions"). In an analysis-driven environment, this is what we usually prefer. However, in certain production settings, you don't want the exception to be raised and rather return the original object (`errors='ignore'`).

# In[ ]:


household['weekday'] = household['weekday'].astype('category', errors='raise')
household.dtypes


# Go ahead and perform the other conversions in the following cell. When you're done, use `dtypes` to check that you have the categorical columns stored as `category`.

# In[ ]:


## Your code below


## -- Solution code


# ### Alternative Solutions (optional)

# In[ ]:


household.select_dtypes(exclude='object').head()


# In[ ]:


pd.concat([
    household.select_dtypes(exclude='object'),
    household.select_dtypes(include='object').apply(
        pd.Series.astype, dtype='category'
    )
], axis=1).dtypes


# In[ ]:


objectcols = household.select_dtypes(include='object')
household[objectcols.columns] = objectcols.apply(lambda x: x.astype('category'))
household.head()


# In[ ]:


household.dtypes


# # Contingency Tables
# 
# One of the simplest EDA toolkit is the frequency table (contingency tables) and cross-tabulation tables. It is highly familiar, convenient, and practical for a wide array of statistical tasks. The simplest form of a table is to display counts of a `categorical` column. Let's start by reading our dataset in; Create a new cell and peek at the first few rows of the data.

# In[ ]:


household = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/household.csv")
household.shape


# In[ ]:


## Your code below


## -- Solution code


# In `pandas`, each column of a `DataFrame` is a `Series`. To get the counts of each unique levels in a categorical column, we can use `.value_counts()`. The resulting object is a `Series` and in descending order so that the most frequent element is on top. 
# 
# Try and perform `.value_counts()` on the `format` column, adding either:
# 
# - `sort=False` as a parameter to prevent any sorting of elements, or
# - `ascending=True` as a parameter to sort in ascending order instead

# In[ ]:


household.sub_category.value_counts(sort=False, ascending=True)


# In[ ]:


## Your code below


## -- Solution code


# `crosstab` is a very versatile solution to producing frequency tables on a `DataFrame` object. Its utility really goes further than that but we'll start with a simple use-case.
# 
# Consider the following code: we use `pd.crosstab()` passing in the values to group by in the rows (`index`) and columns (`columns`) respectively. 

# In[ ]:


pd.crosstab(index=household['sub_category'], columns="count")


# Realize that in the code above, we're setting the row (index) to be `sub_category` and the function will by default compute a frequency table. 

# In[ ]:


pd.crosstab(index=household['sub_category'], columns="count", normalize='columns')


# In the cell above, we set the values to be normalized over each columns, and this will divide each values in place over the sum of all values. This is equivalent to a manual calculation:

# In[ ]:


catego = pd.crosstab(index=household['sub_category'], columns="count")
catego / catego.sum()


# We can also use the same `crosstab` method to compute a cross-tabulation of two factors. In the following cell, the `index` references the sub-category column while the `columns` references the format column:

# In[ ]:


pd.crosstab(index=household['sub_category'], columns=household['format'])


# In[ ]:


household.head()


# This is intuitive in a way: We use `crosstab()` which, we recall, computes the count and we pass in `index` and `columns` which correspond to the row and column respectively.
# 
# When we add `margins=True` to our method call, then an extra row and column of margins (subtotals) will be included in the output:

# In[ ]:


pd.crosstab(index=household['sub_category'], 
            columns=household['format'], 
            margins=True)


# In the following cell, use `pd.crosstab()` with `yearmonth` as the row and `format` as the column. Set `margins=True` to get a total across the row and columns. 

# In[ ]:


## Your code below


## -- Solution code


# If you want an extra challenge, try and modify your code above to include a `normalize` parameter. 
# 
# `normalize` accepts a boolean value, or one of `all`, `index` or `columns`. Since we want it to normalize across each row, we will set this parameter to the value of `index`.

# ## Aggregation Table
# 
# In the following section, we will introduce another parameter to perform aggregation on our table. The `aggfunc` parameter when present, required the `values` parameter to be specified as well. `values` is the values to aggregate according to the factors in our index and columns:

# In[ ]:


pd.crosstab(index=household['sub_category'], 
            columns='mean', 
            values=household['unit_price'],
            aggfunc='mean')


# **Knowledge Check**: Cross tabulation  
# 
# Create a cross-tab using `sub_category` as the index (row) and `format` as the column. Fill the values with the median of `unit_price` across each row and column. Add a subtotal to both the row and column by setting `margins=True`.
# 
# 1. On average, Sugar is cheapest at...?
# 2. On average, Detergent is most expensive at...?
# 
# Create a new cell for your code and answer the questions above.

# In[ ]:


pd.crosstab(index=household['sub_category'],
           columns=household['format'],
           values=household['unit_price'],
           aggfunc='median', margins=True)


# In[ ]:


## Your code below


## -- Solution code


# Reference answer:
# 
# ```
# pd.crosstab(index=household['sub_category'], 
#             columns=household['format'], 
#             values=household['unit_price'],
#             aggfunc='median', margins=True)
# ```

# ### Higher-dimensional Tables
# 
# If we need to inspect our data in higher resolution, we can create cross-tabulation using more than one factor. This allows us to yield insights on a more granular level yet have our output remain relatively compact and structured:

# In[ ]:


pd.crosstab(index=household['yearmonth'], 
            columns=[household['format'], household['sub_category']], 
            values=household['unit_price'],
            aggfunc='median')


# In `pandas` we call a higher-dimensional tables as Multi-Index Dataframe. We are going to dive deeper into the structure of the object on the the next chapter.

# ## Pivot Tables
# 
# If our data is already in a `DataFrame` format, using `pd.pivot_table` can sometimes be more convenient compared to a `pd.crosstab`. 
# 
# Fortunately, much of the parameters in a `pivot_table()` function is the same as `pd.crosstab()`. The noticable difference is the use of an additional `data` parameter, which allow us to specify the `DataFrame` that is used to construct the pivot table.
# 
# We create a `pivot_table` by passing in the following:
# - `data`: our `DataFrame`
# - `index`: the column to be used as rows
# - `columns`: the column to be used as columns
# - `values`: the values used to fill in the table
# - `aggfunc`: the aggregation function

# In[ ]:


pd.pivot_table(
    data=household,
    index='yearmonth',
    columns=['format','sub_category'],
    values='unit_price',
    aggfunc='median'
)


# In[ ]:


pd.pivot_table(
    data=household, 
    index='sub_category',
    columns='yearmonth',
    values='quantity'
)


# A key difference between `crosstab` and `pivot_table` is that `crosstab` uses `len` (or `count`) as the default aggregation function while `pivot_table` using the mean. Copy the cdoe from the cell above and make a change: use `sum` as the aggregation function instead: 

# In[ ]:


## Your code below


## -- Solution code


# # Missing Values and Duplicates
# 
# During the data exploration and preparation phase, it is likely we come across some problematic details in our data. This could be the value of _-1_ for the _age_ column, a value of _blank_ for the _customer segment_ column, or a value of _None_ for the _loan duration_ column. All of these are examples of "untidy" data, which is rather common depending on the data collection and recording process in a company.
# 
# In `pandas`, we use `NaN` (not a number) to denote missing data; The equivalent for datetime is `NaT` but both are essentially compatible with each other. From the docs:
# > The choice of using `NaN` internally to denote missing data was largely for simplicity and performance reasons. We are hopeful that NumPy will soon be able to provide a native NA type solution (similar to R) performant enough to be used in pandas.

# In[ ]:


import math
x=[i for i in range(32000000, 32000005)]
x.insert(2,32030785)
x


# In[ ]:


import math
x=[i for i in range(32000000, 32000005)]
x.insert(2,32030785)

household2 = household.head(6).copy()
household2 = household2.reindex(x)
household2 = pd.concat([household2, household.head(14)])
household2.loc[31885876, "weekday"] = math.nan
household2.iloc[2:8,]


# In the cell above, I used `reindex` to "inject" some rows where values don't exist (receipts item id 32000000 through 32000004) and also set `math.nan` on one of the values for `weekday`. Notice from the output that between row 3 to 8 there are at least a few rows with missing data. We can use `isna()` and `notna()` to detect missing values. An example code is as below:

# In[ ]:


household2['weekday'].isna()


# A common way of using the `.isna()` method is to combine it with the subsetting methods we've learned in previous lessons:

# In[ ]:


household2[household2['weekday'].isna()]


# Go ahead and use `notna()` to extract all the rows where `weekday` column is not missing:

# In[ ]:


## Your code below


## -- Solution code


# Another common use-case in missing values treatment is to count the number of `NAs` across each column:

# In[ ]:


household2.isna().sum()


# When we are certain that the rows with `NA`s can be safely dropped, we can use `dropna()`, optionally specifying a threshold. By default, this method drops the row if any NA value is present (`how='any'`), but it can be set to do this only when all values are NA in that row (`how='all'`).
# 
# ```
#     # drops row if all values are NA
#     household2.dropna(how='all')
#     
#     # drops row if it doesn't have at least 5 non-NA values
#     household2.dropna(thresh=5) 
# ```

# In[ ]:


household2.dropna(thresh=6).head()


# When we have data where duplicated observations are recorded, we can use `.drop_duplicates()` specifying whether the first occurence or the last should be kept:

# In[ ]:


print(household2.shape)
print(household2.drop_duplicates(keep="first").shape)


# **Knowledge Check:** Duplicates and Missing Value  
# _Est. Time required: 20 minutes_
# 
# 1. Duplicates may mean a different thing from a data point-of-view and a business analyst's point-of-view. You want to be extra careful about whether the duplicates is an intended characteristic of your data, or whether it poses a violation to the business logic. 
# 
#     - a. A medical center collects anonymized heart rate monitoring data from patients. It has duplicate observations collected across a span of 3 months
#     - b. An insurance company uses machine learning to deliver dynamic pricing to its customers. Each row contains the customer's name, occupation / profession and historical health data. It has duplicate observations collected across a span of 3 months
#     - c. On our original `household` data, check for duplicate observations. Would you have drop the duplicated rows?
# 
# ---
# 
# 2. Once you've identified the missing values, there are 3 common ways to deal with it:
# 
#     - a. Use `dropna` with a reasonable threshold to remove any rows that contain too little values rendering it unhelpful to your analysis
#     - b. Replace the missing values with a central value (mean or median)
#     - c. Imputation through a predictive model
#         - In a dataframe where `salary` is missing but the bank has data about the customer's occupation / profession, years of experience, years of education, seniority level, age, and industry, then a machine learning model such as regression or nearest neighbor can offer a viable alternative to the mean imputation approach
#  
# Going back to `household2`: what is a reasonable strategy? List them down or in pseudo-code.

# In[ ]:


## Your code below


## -- Solution code


# ## Missing Values Treatment
# 
# Some common methods when working with missing values are demonstrated in the following section. We make a copy of the NA-included DataFrame, and name it `household3`:

# In[ ]:


household3 = household2.copy()
household3.head()


# In the following cell, the technique is demonstrably repetitive or even verbose. This is done to give us an idea of all the different options we can pick from. 
# 
# You may observe, for example that the two lines of code are functionally identical:
# - `.fillna(0)`
# - `.replace(np.nan, 0)`

# In[ ]:


# convert NA categories to 'Missing'
household3[['category', 'format','discount']] = household3[['category', 'format','discount']].fillna('Missing')

# convert NA unit_price to 0
household3.unit_price = household3.unit_price.fillna(0)

# convert NA purchase_time with 'bfill'
household3.purchase_time = household3.fillna(method='bfill')
household3.purchase_time = pd.to_datetime(household3.purchase_time)

# convert NA weekday
household3.weekday = household3.purchase_time.dt.weekday_name

# convert NA quantity with -1
household3.quantity = household3.quantity.replace(np.nan, -1)

household3.head()

