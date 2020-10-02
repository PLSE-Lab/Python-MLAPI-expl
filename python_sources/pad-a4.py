#!/usr/bin/env python
# coding: utf-8

# ## Assignment 4: Clean Walmart Data Further
# 
# You will continue to clean the Walmart data that you worked with last time. A dataset that represents the semiclean data result from Assignment 3 is provided to you. Follow the instructions below to further clean the Walmart data.

# In[ ]:


import pandas as pd


# In[ ]:


# Import semi clean walmart data
# Hint: The name of the file is located in the Workspace section in the right navigation bar in the Kaggle interface.


# In[ ]:


# Check the data types for the imported data frame


# In[ ]:


# Lower case all column values
# Rename IsHoliday column to 'holiday': Refer to the documentation on renaming columns: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html


# In[ ]:


# Convert Yes and No values to True/False Boolean values


# In[ ]:


# Convert Dept to integer


# In[ ]:


# Convert weekly_sales to float:
# Hint: Before converting to float, make sure there are only numbers that are in the string!


# In[ ]:


# Convert date to actual datetime.datetime objects
# Hint: Use the helper function in Walkthrough 4-2 that was provided to you


# In[ ]:


# Make sure all values in date column conform to what the helper function expects


# In[ ]:


# Label the rows with date value of 0 to be "Invalid Date"


# In[ ]:


# Save the Invalid Date rows to a separate dataframe for future analysis. Make to sure to copy the DataFrame to prevent future reference issues


# Get rid of the Invalid Date rows in the clean_df DataFrame. Make to sure to copy the DataFrame to prevent future reference issues


# In[ ]:


# Convert the dates - This may take around ~10 seconds


# In[ ]:


# Confirm that all data has been cleaned and converted to the proper date types
display(clean_df.info()) # display function allows you to display multiple pretty outputs in one cell
display(clean_df.head())

