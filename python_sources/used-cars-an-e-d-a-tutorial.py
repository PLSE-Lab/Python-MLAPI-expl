#!/usr/bin/env python
# coding: utf-8

# # What is Pandas?
# * It is an exploratory data analysis tool which is somewhat similar to Microsoft Excel, but impertively more powerful as it is done in code.
#     *   Pre-requisites to understanding Pandas is a minor experience in object-oriented programming(OOP)
# 
# ## Our Goal
# * The Aim of this tutorial is to provide you a "*crash course"* understanding of the Pandas module without losing focus of another key target, [**exploring data of used cars**](http://www.kaggle.com/lepchenkov/usedcarscatalog)
# 
# #### *** Note :- Kindly Click On The Blue Hyperlinks for the documentations behind the command/datasets.***

# ## Introduction
# * ### Firstly to understand Data Analysis in Kaggle, we need to understand what Kaggle is!
#     * #### Kaggle is a data exploratory website which contains Jupyter Notebooks and Data Sets, a key essential tool in using pandas.
#     * #### This is a Jupyter Notebook inside Kaggle, to access Jupyter Notebook in your own computer without       assistance from Kaggle [take a look at this link](http://jupyter.org/documentation)
# 
# ## Now! The Code      
#    * #### As the data sets we want to access are saved in a different folder inside Kaggle the above command  outputs the file to access the data sets. Along with that using python it imports pandas and Numpy Array into this notebook.
# 
# ##### Note:- Jupyter Notebooks consists of Cells to write Code or Text. A Cell with text similar to this one is called a [*Markdown Cell*](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html) , and a cell with code is called [*Code Cell*](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Display
# ### The above command **imports** the file from the Kaggle folder. We use the [**.head()**](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html) command to show the first 5 rows of the file. Similarly, we can use [.tail()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tail.html) command to show the last 5 rows of the file.
# * ### The () can be used to increase the number of rows we want to see. 
#     * #### Eg:- cars.head(10) shows 10 rows from the file which is assigned to the variable cars       

# In[ ]:


#THIS IS A CODE CELL
cars = pd.read_csv('/kaggle/input/usedcarscatalog/cars.csv')
# Loads input into the variable "cars"
cars.head(10)
#Shows First 10 rows of cars


# ## Data Attributes
# #### As We Would Want to know the number of rows and coloumns present in the file , we use the .shape [command](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html) to see these.
# * ##### The result is displayed as (# Of Rows,# Of Columns)

# In[ ]:


cars.shape


# ## Thoughts Before Initial Analysis
# ### Firstly, I Wanna check whether different companies leads to different lead time in selling the cars.
# 
# ### I'm assuming that "duration_listed" shows the amount of days it was listed before the car was sold.

# In[ ]:


duration = cars.groupby("manufacturer_name").duration_listed.mean().sort_values(ascending = False).reset_index()
# Groups the cars table and finds the average of duration_listed
# Sorts Value By Descending and resets index for ease of *Merge* later into the Data Analysis
# Assigns to variable duration


duration.set_index("manufacturer_name").plot.bar(figsize=(15,5)).set_ylabel("Average Duration Listed")
#Sets the index of the "Duration" Dataseries and plots a bar graph


# ## Split-Apply-Combine,Sorting,Graphing
# ### [.groupby()](http://https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) groups the file by the row specified in the brackets().
# * #### Here we used **.groupby** in manufacturer_names to find the average days(using the [**.mean()**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html) function) the car of the given company was listed before it was sold. 
# * [**.sort_values()**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html) are used to sort the values in this case, days_listed in ascending order.
# * **[.plot](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html).bar()** is used to plot A bar graph
#     * As the default size may not fit a large amount of data, we can use .plot.bar**(figsize = (15,5))** to extend the length of the graph
# .    
# 
# ### Our Analysis
# * #### As we can see from the output above it seems as the car becomes more luxorious  or if the car is not a common car within the region. It takes more time to get sold
# 
# ### Further Thoughts
# * #### We can further extend our analysis by counting the number of features in the car. 
# * #### As price is not provided we can assume more features points towards a luxurious car which means a longer lead time

# ## Our Analysis
# 
# ### **Method Used**
# * #### Firstly, We created a coloumn called Total Features which created the sum of all features(True = 1, False = 0)
# * #### Using .groupby() on Manufacturers_Name(**AGAIN!**), we will find the average features present in the car by Manufacturers Name
# * #### We Merged the Duration_listed and Total_Features and sorted the Total_Features in descending order to see whether more features leads to longer lead time.
#     * #### This was done using **.merge** function which uses manufacturers_name as the common column to merge these two variables together. For More Information on how this is done [refer to this link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)
# 
# * #### [**.reset_index()**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html) removes the current index.

# In[ ]:


cars["total_features"] = cars.feature_1 + cars.feature_2 + cars.feature_3 + cars.feature_4 + cars.feature_5 + cars.feature_6 + cars.feature_7 + cars.feature_8 + cars.feature_9
# Created new coloumn "total_features"
# Sum[] does not work as pandas recognizing it as string instead of integer
#reassigns it back to cars

features = cars.groupby("manufacturer_name").total_features.mean().reset_index()
# cars grouped by manufacturer_name and selects average of total_features, outputs it as Dataseries
# Assigns back to variable "features"

duration = duration.merge(features,left_on="manufacturer_name",right_on="manufacturer_name").set_index("manufacturer_name").sort_values("total_features",ascending = False)
duration
# Merges both the "duration" Dataseries and "features" Dataseries using common index "manufacturer_name"
# Sorts the merged index and assigns it back to duration


# ### Further Analysis
# From this result, we can see that on average, **as more features increased, the lead time to sell the car also increased(checking duration_listed)**. This may be due to an increase in price as the number of features increased. There were also a couple of outliers where very less features also lead to longer lead time.
# 
# ### Further Thoughts
# #### We had found that more features lead to longer lead time for sales,hence, we also would like to check whether our assumption is true by checking the number of listings for the product.
# ####        *  As more features leads to higher prices, we can assume the listings for the product would be less as well.

# In[ ]:


cars.groupby("manufacturer_name").size().sort_values().iloc[0:15].to_frame().rename(columns={0:"num_of_listings"})
# Groups the variable manufacturer_name and counts the row
# Sorts the row according to count and selects the first 15 indices
#resets the series output to Dataframe and converts column name "0" to "num_of_listing"


# ## Frequency Counts, Selection
# ### Methods
# ####            * We use the .groupby() on manufacturer_name and use the [.size()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.GroupBy.size.html) which counts the number of times the manufacturer_name has been mentioned in the row.
# ####            * We then sorted the value in ascending order and used [.iloc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html)[], to show the row indices from 0 to 15(excluding)
# ####            * [**.to_frame()**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_frame.html) converts the series output to dataframe output
# ###        *  Analysis
# ####            * As seen in the result we can come under the consensus that the demand for the product is also lower as the increase in features may lead to higher prices which leads to lesser people purchasing the car and concurrently selling the product too.            
# 

# ## Final Analysis
# * #### We have concluded from our initial analysis through the use of various functions that as features of the vehicle increases the listing days before the vehichle is sold increases.
# * #### We can further investigate the increase in listing days by checking the odometer_value, year_produced,engine_has_gas and transmission. Through checking these additional factors by grouping with the manufacturer_names will help us provide greater information in checking whether these factors would cause an increase in lead time as well.
# * #### I'm interested in investigating engine_has_gas and engine_fuel as I believe this marketing play for engine_has_gas or technology advancements such as Hybrid Types for vehicles with high features in engine_fuel may increase the demand for low-demand vehicles.
