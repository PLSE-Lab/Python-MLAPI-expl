#!/usr/bin/env python
# coding: utf-8

# # Questions
# 1. Find top 5 customers in 2015
# 2. Find top customers in 2014 that did not buy in 2015
# 3. Select details(all columns) for a given set of customers
# 4. Select 2015 transaction amount and number of transcactions for the same customers

# # Pandas DataFrames Basics

# In[ ]:


#Import Package
import pandas as pd


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Import Dataset
customers = pd.read_csv('/kaggle/input/retail-data-customer-summary-learn-pandas-basics/Retail_Data_Customers_Summary.csv')


# ## Find top 5 customers in 2015
# **Concepts covered:**
# 1. head() 
# 2. sort()

# In[ ]:


#the Dataframe name "customers" returns first 5 and last 5 rows
customers


# ### head()

# In[ ]:


#If you use head(), it returns the top 5 rows
customers.head()


# ### sort()

# In[ ]:


#To find the top 5 customers, we need to sort by tran_amount_2015 in descending order
customers.sort_values(by = ['tran_amount_2015'], ascending = False).head()


# ## Find top customers in 2014 that did not buy in 2015
# **Concepts covered: Slicing Dataframe**
# 1. Selecting Single Columns/Multiple Columns
# 2. Sort when data contains NA (na_position argument)

# ### Select Single Column

# In[ ]:


#To find the top 5 customers, we need to sort by tran_amount_2014 in descending order
#Then look at the tran_amount_2015 colums to identify those who did not buy
customers.sort_values(by = ['tran_amount_2014'], ascending = False).head()


# In[ ]:


#We will store in a new dataframe name
customers_2014 = customers.sort_values(by = ['tran_amount_2014'], ascending = False).head()


# ### Select Multiple Column

# In[ ]:


#Then we will select the relevent columns to answer the question
customers_2014[['customer_id','tran_amount_2014','tran_amount_2015']]


# ### Sort when data contains "NA"

# In[ ]:


#Then we ### Select Multiple Columnwill select the relevent columns to answer the question
customers_2014[['customer_id','tran_amount_2014','tran_amount_2015']].sort_values(by = 'tran_amount_2015', na_position = 'first')


# ## Select details(all columns) for a given set of customers
# **Concepts covered:**
# 1. Indexing Column
# 2. Slicing Dataframe 
# 3. Selecting Single Row/Multiple Rows (.loc())
# 4. Making Permanent change to existing dataframe (inplace = True argument)

# **Sample Customers:** (CS4074, CS5057, CS2945, CS4798, CS4424)

# ### Understanding Basics

# In[ ]:


#Returns as error as this id is not present in any column name
customers['CS4074']


# In[ ]:


#so we use .loc to slice/select data by rows
#Still gives an error, as it is unable to find this id in a row (but which row?)
customers.loc['CS4074']


# In[ ]:


#The default index is 0,1,2,3,4,5 ...
customers.head()


# ### Slicing Rows using (.loc())

# In[ ]:


#We can select one row by mentioning the index number
customers.loc[0]


# ### Indexing existing DataFrame

# In[ ]:


#We need to set index before we could easily select rows (just like setting a primary key in SQL)
customers.set_index('customer_id') 


# In[ ]:


#When we see above the index is set. However, when we check its reset to default
customers.head()


# ### Making change permanent (inplace = True)

# In[ ]:


#If you want to make permanent change to the dataframe, we use the inplace argument
customers.set_index('customer_id', inplace = True) 


# In[ ]:


#Now the change is permanent
customers.head()


# **List of Customers:** (CS4074, CS5057, CS2945, CS4798, CS4424)

# In[ ]:


#Now if look for details of a customer, we will be able to find it
customers.loc['CS4074']


# ### Multiple rows selection .loc

# In[ ]:


#To get details of 5 customers, we use a list 
customers.loc[['CS4074', 'CS5057', 'CS2945', 'CS4798', 'CS4424']]


# ## Select 2015 transaction amount and number of transcactions for the same customers
# 1. Selecting multiple columns and rows using (.loc())
# 2. Erasing index for a DataFrame (reset_index())

# ### Selecting multiple columns and rows using (.loc())

# In[ ]:


#We can extend .loc to select both rows and columns (begin with list customers from rows then select columns)
customers.loc[['CS4074', 'CS5057', 'CS2945', 'CS4798', 'CS4424'],['tran_amount_2015','transactions_2015']]


# ### Erasing index for a DataFrame (reset_index())

# In[ ]:


customers.reset_index(inplace = True)


# In[ ]:


#We can no longer select customers using (.loc())
customers.loc[['CS4074', 'CS5057', 'CS2945', 'CS4798', 'CS4424'],['tran_amount_2015','transactions_2015']]


# In[ ]:


customers.head()


# # END
# **Pandas Concepts Covered:**
# 1. head()
# 2. sort()
# 3. Set Index
# 4. Slicing/Selecting DataFrames (columns, rows using .loc)
# 5. Argument (inplace = True)
# 6. Resetting index

# In[ ]:




