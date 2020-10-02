#!/usr/bin/env python
# coding: utf-8

# ## Exercise notebook for the second session (60 min)
# 
# The exercise notebook involves some new concepts not covered in the guided session. You are encouraged to work in groups of 2-4 if that helps to speed things up. Please ask for help from the instructor and/or TAs. The session is time-bound, so make sure you are not stuck at a problem for too long before asking for help.  

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter('ignore')


# ### 1. Feature engineering on the Titanic dataset to create a new column for group size (30 min):
# Loading the [Titanic dataset](https://www.kaggle.com/c/titanic):

# In[ ]:


path = '../input/titanic/'
train = pd.read_csv(path + 'train.csv') 
test = pd.read_csv(path + 'test.csv') 


# Create a DataFrame named `df` by concatenating the `train` and `test` datasets one below the other. Hint: Use `concat()`.

# In[ ]:


df = pd.concat([train, test])


# In[ ]:


df.head()


# Create a new column named *Family* by adding the columns *SibSp* and *Parch* and then add 1 to it. Note: [Here](https://www.kaggle.com/c/titanic/data) is the description for features *SibSp* and *Parch*.

# In[ ]:


df['Family'] = df['SibSp'] + df['Parch'] + 1
df.head()


# Now we check the survival rates with respect to the family size.

# In[ ]:


sns.barplot(x='Family', y='Survived', data=df);


# Some passengers that appear to be traveling alone by account of their family size were part of a group traveling on the same ticket. To see this, get all the passengers traveling on the ticket "1601" (there are 8 of them).

# In[ ]:





# One can check that there are many tickets shared among passengers that may or may not be family members.

# In[ ]:


df['Ticket'].value_counts()[:15]


# Create a new column named *TicketCount* that counts the total number of passengers traveling in each passengers' ticket.
# 
# Hint: 
# - First group passengers based on their tickets using `groupby()` on the *Ticket* column.
# - For the grouped object, pick any column.
# - Use `transform()` for this unique identifier column with the function `"count"` to create a new column *TicketCount*.
# 
# For example, we created *MedianAge* using the following code:   
# ```df['MedianAge'] = df.groupby('Title')['Age'].transform("median")```

# In[ ]:





# In[ ]:


df.head()


# Plot the survival rates based on the *TicketCount* using `sns.barplot()` (see above).

# In[ ]:





# It does seem that the number of co-travelers have an impact on the survival rates.

# Create a new column named *GroupSize* by picking the maximum value among the columns *Family* and *TicketCount*.   
# Note: We consider groups to be either family members or those traveling on the same ticket.   
# Hint: Use built-in `max()` function for pandas on the two relevant columns with the appropriate value for the `axis` parameter. 

# In[ ]:





# Plot the survival rates based on the *GroupSize* using `sns.barplot()`.

# In[ ]:





# Check the number of rows where *Groupsize* is not equal to *Family*. Similarly, check the number of rows where *TicketCount* is not equal to *Family*.

# In[ ]:


df[df['GroupSize'] != df['Family']].shape[0], 
df[df['GroupSize'] != df['TicketCount']].shape[0]


# The output must be `(200, 84)`. Check your above code, if you get a different output.

# ### 2. Creating a new column using grouping on two columns of [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) dataset (10 min):

# In[ ]:


path = '../input/competitive-data-science-predict-future-sales/'
df = pd.read_csv(path + 'sales_train.csv')
df.head()


# [The description](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data) for each of the following columns is as follows. 
# - date: date in format dd/mm/yyyy
# - date_block_num: a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# - shop_id: unique identifier of a shop
# - item_id: unique identifier of a product
# - item_price: current price of an item
# - item_cnt_day: number of products sold
#        
# The aim of the competition is to predict the monthly sales. We are given the daily sales of items in each shop in the column *item_cnt_day*. We want to create a new column named *item_cnt_monthly* that gives the monthly sales of the items using the columns *item_cnt_day* and *date_block_num*. 

# Currently, the rows are sorted based on the *date* column. We want them sorted first w.r.t. *date_block_num* and then  w.r.t. *item_id* within each *date_block_num*. Hint: Use `sort_values()` on the two columns: `'date_block_num','item_id'`

# In[ ]:





# We sorted the rows so that it will make it easier to check whether our code below is working correctly. For the *date_block_num* equal to 0, we can see that the item with *item_id* 19 is sold once, one with *item_id* 27 is sold 7 times and so on.

# In[ ]:


df.head(20)


# We can use `groupby()` on the two columns - *date_block_num* and *item_id* and sum the values in the *item_cnt_day* column in each grouping as follows.

# In[ ]:


df.groupby(['date_block_num', 'item_id'])['item_cnt_day'].sum()


# Create a new column named *item_cnt_monthly* that gives the monthly sales of the items
# 
# Hint: Use `transform()` with the function `"sum"` along with `groupby()`.

# In[ ]:





# Check that the code indeed worked as expected.

# In[ ]:


df.head(25)


# ### 3. Merging columns from different datasets (20 min):

# A simple illustration to merge two datasets using `merge()`

# In[ ]:


df1 = pd.DataFrame({'CourseCode': ['PHYS024', 'CSCI35', 'ENGR156'], 
                   'CourseName': ['Mechanics and Wave Motion', 
                                  'Computer Science for Insight',
                                 'Intro to Comm & Info Theory']})

df2 = pd.DataFrame({'Professor': ['Zachary Dodds', 'Vatche Sahakian', 
                                  'Timothy Tsai', 'Brian Shuve'],
                    'CourseCode': ['CSCI35', 'PHYS024',  'ENGR156', 'PHYS024']})

df1.head()


# In[ ]:


df2.head()


# In[ ]:


pd.merge(df2, df1)


# Please refer to the documents [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html) and [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) to better grasp how and when to use `merge()` function. 

# #### Merging dataframes from [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data) dataset:
# Load the four files into separate dataframes:   
# `aisles.csv`   
# `departments.csv`  
# `products.csv`  
# `order_products__train.csv`

# In[ ]:





# Familiarize yourself with the dataframes. Hint: Use `head()`.  
# Note: You might want to add code cells. Please ask for help unless you already know how to add a code cell.

# In[ ]:





# Get a dataframe consisting of the name of the products along with their aisle names and department names for the order with `order_id` equal to 1.   
# Note: dataframe must have ***8 rows and only three columns:
# ```'product_name', 'aisle', 'department'```***
# 
# Hint: 
# - First slice out 8 rows from the order_products dataframe that corresponds with `order_id` equal to 1 and save it in a dataframe.
# - Use `merge()` multiple times to merge two dataframes at a time starting with the dataset that have a column in common with the dataframe corresponding to 'order_products__train.csv'.
# - Finally slice out the 3 selective columns: `'product_name', 'aisle', 'department'`

# In[ ]:





# In[ ]:


df.shape


# The output should be (8, 3). Please check your code above if you get something else.

# #### Exploring csv files (optional)
# 
# Dataframes are easily convertible to and from csv (comma seperated text) files. Convert the last dataframe to a csv file using `to_csv()`.

# In[ ]:





# Instructions for the next steps if using Kaggle kernel:
# * Commit the kernel so that the output file is generated. You can visit the kernels page (without the "/edit" and instead use "/output" at the end) in the link to download the output file. Please ask for help if the instructions are not clear. 
# * Open the csv file using MS-Excel, any text editor or other applications of your choice. 
# * Edit the file, save it and upload it to the Kaggle kernel using 'ADD DATASET' option.
# 
# Instructions for the next steps if using Jupyter Notebook in your laptop:
# * Go to the folder where Notebook is saved and locate the output file.
# * Open the csv file using MS-Excel, any text editor or other application of your choice that supports csv files. 
# * Edit the file and save it.

# Now, read the csv file and check whether the changes are reflected.

# ### Acknowledgement:
# The following datasets openly available in Kaggle are used in the exercises.
# * [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic)
# * [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
# * [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)
# 

# ### Next step:
# 
# Please pick one or more datasets to explore. Suggestions are given [here](http://www.aashitak.com/ML-Workshops/).
# 
# 
# **Note:**
# The solutions for this exercise can be found [here](https://github.com/AashitaK/ML-Workshops/blob/master/Session%202/Exercise%202%20with%20solutions.ipynb)

# In[ ]:




