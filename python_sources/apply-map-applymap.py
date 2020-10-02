#!/usr/bin/env python
# coding: utf-8

# # Apply, Map and Applymap
# ### A simple explanation with examples to illustrate the use and differences

# #### Load libraries and create data to be used

# In[ ]:


import numpy as np
import pandas as pd

data = {'units': [1,2,3,4,5,6,7,8,9],
        'tens':['10','20','30','40','50','60','70','80','90'],
        'hundreds': [100.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0,900.0]
       }
dict = {1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine'}

df = pd.DataFrame(data)
df


# > ### <b><i>Map</i></b>: Works only on Series and can be used to map every elemnt of the series by a <i>function</i> or by a <i>dictionary</i>

# In[ ]:


# Add a new column for the units in words and use the dict to map it from the units columns
# Here we use a dictionary to may the 'units' column to its value in words and create the 
# 'units_in_words' column
df['units_in_words']=df['units'].map(dict)
df


# In[ ]:


# Use a function in map to get the length of the 'units_in_words'
# Here we are using the in-built function len, but user defined functsions can also be used
df['len_units_in_words']=df['units_in_words'].map(len)
df


# ### <i><b>Apply</b></i> : works on both - Series as well as DataFrames
# ***Note: Only a function can be used with Apply; Dictionary cannot be used***

# #### Let' use apply on a Series

# In[ ]:


# Can we use a dictionary to map a column?
# Below line of code throws as error
# df['words'] = df['units'].apply(dict) 


# In[ ]:


# Apply can only use a function
# We use a lambda function and apply it to the 'units' columns to create the 'squares' column
df['squares']=df['units'].apply(lambda x: x**2)
df


# #### Apply can also be used with a DataFrame. 
# #### In such a case it takes either the whole row or the columns as an unit. Hence, axis parameter needs to be passed to it to indicate whether apply needs to work on rows or columns
# 
# 

# In[ ]:


# Let's use this to add the units and hundres columns

def sum_up(x): # Note only one elemnt is passed. This may be list/array
    return (x[0] + x[1])

def add_up(x1, x2): # Try to pass two elments and add them
    return (x1 + x2)

# Here we apply the sum_up function on the two columns 'units' and 'hundreds' 
# and since the action if column wise axis passed as 1
df['units_plus_hundreds']=df[['units','hundreds']].apply(sum_up,axis=1)
print(df)

# This throws an error
# df['units_plus_hundreds']=df[['units','hundreds']].apply(add_up,axis=1)
# print(df)


# In[ ]:


# Here we use axis=0 and apply the function sum to the columns 'units' and 'hundreds'
df[['units','hundreds']].apply(sum, axis=0)


# ### <i><b>Applymap</b></i> : works only on DataFrame, but unlike Apply it works on each element and hence does not need axis as a parameter

# In[ ]:


# Check the data types before using applymap
df.dtypes


# In[ ]:


# Now let's convert the hundreds and unit_plus_hyndreds fields from float to int
df[['hundreds','units_plus_hundreds']] = df[['hundreds','units_plus_hundreds']].applymap(int)
df.dtypes


# #### Applymap cannot be used to do a mapping using dictionary

# In[ ]:


# This will throw an error
# df['words'] = df[['len_units_in_words']].applymap(dict)
# df


# In[ ]:


# Take sqrt of the squares columns
df['sqrt'] = df[['squares']].applymap(np.sqrt)
df


# #### Hope these examples have helped in understanding Apply, Map and Applymap and their differences

# In[ ]:




