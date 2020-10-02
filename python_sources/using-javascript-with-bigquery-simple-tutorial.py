#!/usr/bin/env python
# coding: utf-8

# ## How to use javascript functions with Big Query
# Getting too much data on your side then doing analysis is going to result in a timeout error or quota exceeded in big query. 
# 
# But there is one feature in Google Big Query which can help us do many complex analysis right through the query and just get the complete result in just one go..
# 
# 
# ## User Defined Functions:
# Google big query gives us user defined functions which we can define in our query very easily,  but there is just one issue, user defined functions are only available in Javascript. 
# Now without any more theorotical writing lets start with the example.

# In[ ]:


import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "github-repos")


# **Lets consider a simple javascript function which takes a string and returns everything within "[" and "]"
# Example: Input: "Hello my name is [ibaaaad]"
# Output: ibaaaad**

# ### The syntax is quite simple: you start with CREATE TEMPORARY FUNCTION which explains itself that you want to create a function for a query. 
# #### Then Write the name of the function and in brackets the arguments
# #### After that with RETURNS keyword you specify the return type of the function
# #### then you tell the big query that what language you are using in this case LANGUAGE JS 
# ### Finally after the AS keyword you start writing your function logic with triple (double) quotes.
# ### """ Function logic"""
# 

# In[ ]:


javascript_query='''
CREATE TEMPORARY FUNCTION 
myFunc(mystring STRING)
RETURNS STRING
LANGUAGE js AS
"""
var start=mystring.indexOf('[');
var end=mystring.indexOf(']');
var res='nothing';
if (start==-1 || end==-1){
    res='nothing';
}else{
 res = mystring.slice(start+1, end);
}

return res;
""";
'''
print (javascript_query)


# ### As we can see this function just takes a string and finds "[" and "]" and returns the substring that occurs between [ and ]
# 

# ### Lets write an sql query and use this function in the sql query.

# In[ ]:


my_sql_query='''SELECT myFunc(names) as EDITED_NAME 
FROM UNNEST(["H[ibbu]ah", "Max", "Jakob","HELLO WORLD[SDSDsdsd]"]) AS names'''


# #### This query will simple get names from an array, and pass it to myFunc the Function we made above and return the result.
# ### Now we will concatenate the function we have created and the sql query and finally send it to big query.
# This is how our final query looks like

# In[ ]:


final_query=javascript_query+my_sql_query
print (final_query)


# In[ ]:


bq_assistant.estimate_query_size(final_query)


# In[ ]:


## Seems like we are good to go
result=bq_assistant.query_to_pandas_safe(final_query)
print (result)


# ### As we can clearly see, We just used a javascript function in our SQL query, We can use more than one functions too.
# ### With these functions we can do complex analysis which we cannot do with standard sql in Big Query.
# #### Moreover we can include whole javascript libraries over here.
# ### To be Continued
# #### Please do give your suggestions, Thank you

# For more information you can see the documentation over here: https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions
