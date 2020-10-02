#!/usr/bin/env python
# coding: utf-8

# # Reshaping Pandas DataFrame

# ## Importing required packages 

# Importing packages needed for this notebook with an alias. We use alias so as to avoid typing name of package multiple times as we use it.

# In[ ]:


import pandas as pd
import numpy as np


# Checking pandas version installed in the system

# In[ ]:


pd.__version__


# Dataframes are very similar to excel tables, and sometimes we might want to look at them in different view/style. Like exchanging columns to rows and vice versa. Let us look at some of the famous reshaping functions available for dataframe.

# I like to think of reshaping functions in two types. First type of functions simple reform the existing dataframe. For example, they change columns to rows and rows to columns. The second type of functions would aggregate the information along with reforming them. In most of the real world cases, we would be using the second type of functions as it would give us a peek into the higher level summaries or aggregation as needed.

# ## Type - 1 : Reforming without aggregation 

# Reforming without aggregation can and should ideally be applied on data where there is a unique combination of selections being made. Otherwise, there is a good chance that they would throw an error for certain functions.

# To explain the reforming without aggregation, we would first declare a dataframe. The declaration and the dataframe would be as follows:

# In[ ]:


np.random.seed(100)
df=pd.DataFrame({"Date":pd.Index(pd.date_range(start='2/2/2019',periods=3)).repeat(3),
             "Class":["1A","2B","3C","1A","2B","3C","1A","2B","3C"],
             "Numbers":np.random.randn(9)})

df['Numbers2'] = df['Numbers'] * 2

df


# ### Pivot 

# Pivot method is typically used to create a pivot style view of data where the users can specify rows (in python it is called index) and columns. These two parameters would give a structure to the view whereas the information to be populated would be from the data that is being used to create pivot. The information can also be selectively populated by using values parameter.

# In[ ]:


df.pivot(index='Date', columns='Class', values='Numbers')


# In[ ]:


df.pivot(index='Date', columns='Class')


# In[ ]:


df.pivot(index='Date', columns='Class')['Numbers']


# In[ ]:


df.pivot(index='Date', columns='Class')['Numbers'].reset_index()


# In[ ]:


np.random.seed(100)
df1=pd.DataFrame({"Date":pd.Index(pd.date_range(start='2/2/2019',periods=3)).repeat(3),
             "Class":["1A","1A","1A","2B","2B","2B","3C","3C","3C"],
             "Numbers":np.random.randn(9)})

df1


# You cannot have pivot done where there are duplicates. You will get an error like shown below.

# In[ ]:


df1.pivot(index='Date', columns='Class')


# ### Melt 

# Melt is a function which is used to convert columns to rows. That means that this function is useful for when the users would like to bring one or more columns information into rows. This function would create two new columns by removing all other columns apart from the ones mentioned in its id_vars parameter and displays the column name in one column and its value in another column.

# In[ ]:


df


# In[ ]:


df.melt(id_vars=['Date','Class'])


# In[ ]:


df.melt(id_vars=['Date','Class'],value_vars=['Numbers'])


# In[ ]:


df.melt(id_vars=['Date','Class'],value_vars=['Numbers'],value_name="Numbers_Value",var_name="Num_Var")


# In[ ]:


df1


# In[ ]:


df1.melt(id_vars=['Date','Class'],value_vars=['Numbers'],value_name="Numbers_Value",var_name="Num_Var")


# ### Stack and Unstack 

# Stack and Unstack perform columns to rows and rows to columns operations respectively. Both these functions are definitely one of the less used functions of reshaping in pandas as one would use pivot to achieve the result they want most of the time and hence it would not be needed.

# #### Stack 

# In[ ]:


df


# In[ ]:


df.set_index(["Date","Class"]).stack()


# In[ ]:


df.set_index(["Date","Class"]).stack(0)


# In[ ]:


df.set_index(["Date","Class"]).stack(-1)


# #### Unstack 

# In[ ]:


df.set_index(["Date","Class"]).stack().unstack()


# In[ ]:


df.set_index(["Date","Class"]).stack().unstack(-1)


# In[ ]:


df.set_index(["Date","Class"]).stack().unstack(0)


# In[ ]:


df.set_index(["Date","Class"]).stack().unstack(1)


# In[ ]:


df.set_index(["Date","Class"]).stack().unstack([1,-1])


# ## Type - 2: Reforming with aggregation

# Unlike the type-1 functions, type-2 functions give an aggregated view of information. These will be very useful in case the users would like to have some type of summary around the data. We will be using the same dataframe that we used for Type -1 functions to look into type -2 functions as well.

# To explain the reforming without aggregation, we would first declare a dataframe. The declaration and the dataframe would be as follows:

# In[ ]:


df=pd.DataFrame({"Date":pd.Index(pd.date_range(start='2/2/2019',periods=2)).repeat(4),
             "Class":["1A","2B","3C","1A","2B","3C","1A","2B"],
             "Numbers":np.random.randn(8)})

df['Numbers2'] = df['Numbers'] * 2

df


# ### Group By

# Group by is the function that I use the more often than any other function mentioned in this article. This is because, it is very intuitive to use and has very useful parameters that can help one to view different aggregations for different columns.

# In[ ]:


df.groupby('Date')["Numbers"].mean()


# In[ ]:


df.groupby('Date',as_index=False)["Numbers"].mean()


# In[ ]:


df.groupby(['Date','Class'],as_index=False)["Numbers"].mean()


# In[ ]:


df.groupby(['Date','Class'],as_index=False)[["Numbers","Numbers2"]].mean()


# In[ ]:


df.groupby(['Date'],as_index=False).aggregate({"Numbers":"sum","Numbers2":"mean"})


# ### Pivot Table 

# Pivot Table functions in the same way that pivots do. However, pivot table has one additional and significant argument/parameter which specifies the aggregation function we will be using to aggregate the data.

# In[ ]:


df.pivot(index="Date",columns="Class")


# In[ ]:


df.pivot_table(index="Date",columns="Class")


# In[ ]:


df.pivot_table(index="Date",columns="Class",aggfunc="sum")


# ### Crosstab 

# The last function in this article would be crosstab. This function by default would give the count or frequency of occurrence between values of two different columns.

# In[ ]:


df


# In[ ]:


pd.crosstab(df.Date,df.Class)


# In[ ]:


pd.crosstab(df.Date,df.Class,values=df.Numbers,aggfunc='sum')


# In[ ]:


pd.crosstab(df.Date,df.Class,values=df.Numbers,aggfunc='mean')


# In[ ]:




