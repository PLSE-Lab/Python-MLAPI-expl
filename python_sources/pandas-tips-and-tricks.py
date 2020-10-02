#!/usr/bin/env python
# coding: utf-8

# # Pandas...............

# ![](https://media.tenor.com/images/90cd700e5b9451af5a803e4441ae80d8/tenor.gif)

# # Load example datasets

# In[ ]:


import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))


# In[ ]:


drinks = pd.read_csv('../input/drinks.csv')


# # Show installed versions
# 
# Sometimes you need to know the pandas version you're using, especially when reading the pandas documentation. You can show the pandas version by typing:
# 

# In[ ]:


pd.__version__


# In[ ]:


pd.show_versions()
#You can see the versions of Python, pandas, NumPy, matplotlib, and more.


# # Create an example DataFrame
# Let's say that you want to demonstrate some pandas code. You need an example DataFrame to work with.
# There are many ways to do this, but my favorite way is to pass a dictionary to the DataFrame constructor, in which the dictionary keys are the column names and the dictionary values are lists of column values:

# In[ ]:


df = pd.DataFrame({'col one':[100, 200], 'col two':[300, 400]})
df


# Now if you need a much larger DataFrame, the above method will require way too much typing. In that case, you can use NumPy's random.rand() function, tell it the number of rows and columns, and pass that to the DataFrame constructor:

# In[ ]:


pd.DataFrame(np.random.rand(4, 8))
#That's pretty good, but if you also want non-numeric column names, you can coerce a string of letters to a list and then pass that list to the columns parameter:


# In[ ]:


pd.DataFrame(np.random.rand(4, 8), columns=list('abcdefgh'))
#As you might guess, your string will need to have the same number of characters as there are columns.


# # Rename columns
# 

# In[ ]:


df


# I prefer to use dot notation to select pandas columns, but that won't work since the column names have spaces. Let's fix this.
# 
# The most flexible method for renaming columns is the rename() method. You pass it a dictionary in which the keys are the old names and the values are the new names, and you also specify the axis:

# In[ ]:


df = df.rename({'col one':'col_one', 'col two':'col_two'}, axis='columns')
print(df)


# The best thing about this method is that you can use it to rename any number of columns, whether it be just one column or all columns.
# Now if you're going to rename all of the columns at once, a simpler method is just to overwrite the columns attribute of the DataFrame:

# In[ ]:


df.columns = ['col_one', 'col_two']


# Now if the only thing you're doing is replacing spaces with underscores, an even better method is to use the str.replace() method, since you don't have to type out all of the column names:

# In[ ]:


df.columns = df.columns.str.replace(' ', '_')


# All three of these methods have the same result, which is to rename the columns so that they don't have any spaces:

# In[ ]:


df


# Finally, if you just need to add a prefix or suffix to all of your column names, you can use the add_prefix() method...

# In[ ]:


df.add_prefix('X_')


# In[ ]:


df.add_suffix('_Y')


# # Reverse row order

# In[ ]:


#Let's take a look at the drinks DataFrame:
drinks.head()


# This is a dataset of average alcohol consumption by country. What if you wanted to reverse the order of the rows?
# 
# The most straightforward method is to use the loc accessor and pass it ::-1, which is the same slicing notation used to reverse a Python list:

# In[ ]:


drinks.loc[::-1].head()


# What if you also wanted to reset the index so that it starts at zero?
# 
# You would use the reset_index() method and tell it to drop the old index entirely:

# In[ ]:


drinks.loc[::-1].reset_index(drop=True).head()


# As you can see, the rows are in reverse order but the index has been reset to the default integer index.

# #  Reverse column order
# 
# Similar to the previous trick, you can also use loc to reverse the left-to-right order of your columns:

# In[ ]:


drinks.loc[:, ::-1].head()


# The colon before the comma means "select all rows", and the ::-1 after the comma means "reverse the columns", which is why "country" is now on the right side.

# # Select columns by data type

# In[ ]:


#Here are the data types of the drinks DataFrame:
drinks.dtypes


# Let's say you need to select only the numeric columns. You can use the select_dtypes() method:

# In[ ]:


drinks.select_dtypes(include='number').head()


# This includes both int and float columns.
# 
# You could also use this method to select just the object columns:

# In[ ]:


drinks.select_dtypes(include='object').head()


# You can tell it to include multiple data types by passing a list:

# In[ ]:


drinks.select_dtypes(include=['number', 'object', 'category', 'datetime']).head()


# You can also tell it to exclude certain data types:

# In[ ]:


drinks.select_dtypes(exclude='number').head()


# # Convert strings to numbers
# Let's create another example DataFrame:

# In[ ]:


df = pd.DataFrame({'col_one':['1.1', '2.2', '3.3'],
                   'col_two':['4.4', '5.5', '6.6'],
                   'col_three':['7.7', '8.8', '-']})
df


# These numbers are actually stored as strings, which results in object columns:

# In[ ]:


df.dtypes


# In order to do mathematical operations on these columns, we need to convert the data types to numeric. You can use the astype() method on the first two columns:

# In[ ]:


df.astype({'col_one':'float', 'col_two':'float'}).dtypes


# However, this would have resulted in an error if you tried to use it on the third column, because that column contains a dash to represent zero and pandas doesn't understand how to handle it.
# 
# Instead, you can use the to_numeric() function on the third column and tell it to convert any invalid input into NaN values:

# In[ ]:


pd.to_numeric(df.col_three, errors='coerce')


# If you know that the NaN values actually represent zeros, you can fill them with zeros using the fillna() method:

# In[ ]:


pd.to_numeric(df.col_three, errors='coerce').fillna(0)


# Finally, you can apply this function to the entire DataFrame all at once by using the apply() method:

# In[ ]:


df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
df


# This one line of code accomplishes our goal, because all of the data types have now been converted to float:

# In[ ]:


df.dtypes

