#!/usr/bin/env python
# coding: utf-8

# # Questions
# 1. Find top 5 customers in 2015
# 2. Find top customers in 2014 that did not buy in 2015
# 3. Select details(all columns) for a given set of customers
# 4. Select 2015 transaction amount and number of transcactions for the same customers

# # Pandas DataFrames Basics

# In[ ]:


#Import Packages


# In[ ]:


#Import Dataset (/kaggle/input/retail-data-customer-summary-learn-pandas-basics/Retail_Data_Customers_Summary.csv)


# ## Find top 5 customers in 2015
# **Concepts covered:**
# 1. head() 
# 2. sort()

# In[ ]:


#the Dataframe name "customers" returns first 5 and last 5 rows


# ### head()

# In[ ]:


#If you use head(), it returns the top 5 rows


# ### sort()

# In[ ]:


#To find the top 5 customers, we need to sort by tran_amount_2015 in descending order


# ## Find top customers in 2014 that did not buy in 2015
# **Concepts covered: Slicing Dataframe**
# 1. Selecting Single Columns/Multiple Columns
# 2. Sort when data contains NA (na_position argument)

# ### Select Single Column

# In[ ]:


#To find the top 5 customers, we need to sort by tran_amount_2014 in descending order
#Then look at the tran_amount_2015 colums to identify those who did not buy


# In[ ]:


#We will store in a new dataframe name


# ### Select Multiple Column

# In[ ]:


#Then we will select the relevent columns to answer the question


# ### Sort when data contains "NA"

# In[ ]:


#Then we ### Select Multiple Columnwill select the relevent columns to answer the question


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


# In[ ]:


#so we use .loc to slice/select data by rows
#Still gives an error, as it is unable to find this id in a row (but which row?)


# In[ ]:


#The default index is 0,1,2,3,4,5 ...


# ### Slicing Rows using (.loc())

# In[ ]:


#We can select one row by mentioning the index number


# ### Indexing existing DataFrame

# In[ ]:


#We need to set index before we could easily select rows (just like setting a primary key in SQL)


# In[ ]:


#When we see above the index is set. However, when we check its reset to default


# ### Making change permanent (inplace = True)

# In[ ]:


#If you want to make permanent change to the dataframe, we use the inplace argument
 


# In[ ]:


#Now the change is permanent


# **List of Customers:** (CS4074, CS5057, CS2945, CS4798, CS4424)

# In[ ]:


#Now if look for details of a customer, we will be able to find it


# ### Multiple rows selection .loc

# In[ ]:


#To get details of 5 customers, we use a list 


# ## Select 2015 transaction amount and number of transcactions for the same customers
# 1. Selecting multiple columns and rows using (.loc())
# 2. Erasing index for a DataFrame (reset_index())

# ### Selecting multiple columns and rows using (.loc())

# In[ ]:


#We can extend .loc to select both rows and columns (begin with list customers from rows then select columns)


# ### Erasing index for a DataFrame (reset_index())

# In[ ]:





# In[ ]:


#We can no longer select customers using (.loc())


# # END
# **Pandas Concepts Covered:**
# 1. head()
# 2. sort()
# 3. Set Index
# 4. Slicing/Selecting DataFrames (columns, rows using .loc)
# 5. Argument (inplace = True)
# 6. Resetting index

# In[ ]:




