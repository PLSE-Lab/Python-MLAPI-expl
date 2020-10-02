#!/usr/bin/env python
# coding: utf-8

# **PANDAS**
# 
# * Pandas is an open source library built on top of Numpy.
# * It allows for fast analysis and data cleaning and preparation.
# * It excels in performance and productivity.
# * It also have build-in visualization features.
# * It can work with data from a wide variety of sources.
# 
# 
# You can use for Anaconda case -  conda install pandas
# 
# Otherwise on cmd prompt - pip install pandas

# **Topic **
# 1. Series
# 2. Data Frames
# 3. Missing Data
# 4. Group by
# 5. Merging, Joining and Concatenation
# 6. Operations
# 7. Data Input and Output

# **1. Series**
# 
# 
# A Series is very similar to a NumPy array (in fact it is built on top of the NumPy array object).
# What differentiates the NumPy array from a Series, is that a Series can 
# have axis labels, meaning it can be indexed by a label, instead of just a number location. 
# It also doesn't need to hold numeric data, it can hold any arbitrary Python Object.
# 

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd


# In[ ]:


labels=['a','b','c']   #list
my_data = [10,20,30]   #list
arr = np.array(my_data)  #use numpy
arr


# In[ ]:


d = {'a':10,'b':20,'c':30}  # dictionary (key -value pair relationship)
# Using List
pd.Series(data = my_data)


# In[ ]:


pd.Series(data =my_data,index =labels)   #Note data comes after index


# In[ ]:


#You can simply this as 
pd.Series(my_data,labels)


# In[ ]:


# Numpy Arrays
pd.Series(arr) 


# In[ ]:


pd.Series(arr,labels)


# In[ ]:


pd.Series(d)


# **Data in Series**
# 
# A pandas Series can hold a variety of object types:

# In[ ]:


pd.Series(data=labels)


# In[ ]:


# Even functions (although unlikely that you will use this)
pd.Series([sum,print,len])


# **Using an Index
# **
# 
# The key to using a Series is understanding its index. Pandas makes use of these index names or numbers by allowing for fast look ups of information (works like a hash table or dictionary).
# 
# Let's see some examples of how to grab information from a Series. Let us create two sereis, ser1 and ser2:
# 

# In[ ]:


ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
ser1


# In[ ]:


ser1['USA']


# In[ ]:


ser2 = pd.Series([1,2,5,4],['USA','Germany','Italy','Japan'])
ser2


# In[ ]:


ser2['USA']


# In[ ]:


# Operations are then also done based off of index:
ser1 + ser2
# Note :  When you add two series, your Integers are converted as float. (Also see some rows got NAN values) 
# Tell what's the reason behind it in the comment below.


# **Data Frames**
# 
# DataFrames are the workhorse of pandas and are directly inspired by the R programming language. We can think of a DataFrame as a bunch of Series objects put together to share the same index. Let's use pandas to explore this topic!

# In[ ]:


#however we have already install above (to remind you I typed again)
import numpy as np
import pandas as pd


# In[ ]:


from numpy.random import randn
np.random.seed(101)


# Docstring:
# seed(seed=None)
# 
# Seed the generator.
# 
# This method is called when `RandomState` is initialized. It can be
# called again to re-seed the generator. For details, see `RandomState`.
# 
# Parameters
# ----------
# seed : int or 1-d array_like, optional
#     Seed for `RandomState`.
#     Must be convertible to 32 bit unsigned integers.
# 
# See Also
# --------
# RandomState
# Type:      builtin_function_or_method

# In[ ]:


df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
df


# ## Selection and Indexing
# 
# Let's learn the various methods to grab data from a DataFrame

# In[ ]:


df['W']


# In[ ]:


type(df['W'])   # Return type


# In[ ]:


# Pass a list of column names
df[['W','Z']]


# In[ ]:


# SQL Syntax (NOT RECOMMENDED!)
df.W


# **Creating a new column:**

# In[ ]:


df['new'] = df['W'] + df['Y']


# In[ ]:


df


# ** Removing Columns**

# In[ ]:


df.drop('new',axis=1)   # Note if you don't specify the axis =1 it will assume to be zero
# axis =1 means for columns
# axis =0 means for rows/index


# In[ ]:


df
#see new is not deleted or drop permanently
# so we have to use inplace = True


# In[ ]:


df.drop('new',axis=1,inplace=True)
df


# **We can also drop Rows in same way**

# In[ ]:


df.drop('E',axis=0)


# **Selecting Rows**

# In[ ]:


df.loc['A']


# In[ ]:


# Or select based off of position instead of label 
df.iloc[2]    # So we will get all values of C row


# ** Selecting subset of rows and columns **

# In[ ]:


df.loc['B','Y']


# In[ ]:


df.loc[['A','B'],['W','Y']]


# ### Conditional Selection
# 
# An important feature of pandas is conditional selection using bracket notation, very similar to numpy:

# In[ ]:


df


# In[ ]:


df>0    # output true(means positive) or false(means negative)


# In[ ]:


df[df>0]
# In this case all the false value is replace by NaN and True value replace by original value.


# In[ ]:


df['W']>0


# In[ ]:


df[df['W']>0]    # element C row did you notice!!!!


# In[ ]:


df[df['W']>0]['Y']
# B and D elemented because of negative value in Y column


# In[ ]:


df[df['W']>0][['Y','X']]


# For two conditions you can use | and & with parenthesis:

# In[ ]:


df[(df['W']>0) & (df['Y'] > 1)]


# ## More Index Details
# 
# Let's discuss some more features of indexing, including resetting the index or setting it something else. We'll also talk about index hierarchy!

# In[ ]:


df


# In[ ]:


# Reset to default 0,1...n index
df.reset_index()


# In[ ]:


newind = 'CA NY WY OR CO'.split()
newind


# In[ ]:


df['States'] = newind
df


# In[ ]:


df.set_index('States')


# In[ ]:


df


# In[ ]:


df.set_index('States',inplace=True)
df


# ## Multi-Index and Index Hierarchy
# 
# Let us go over how to work with Multi-Index, first we'll create a quick example of what a Multi-Indexed DataFrame would look like:

# In[ ]:


# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)


# In[ ]:


hier_index


# In[ ]:


df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df


# Now let's show how to index this! For index hierarchy we use df.loc[], if this was on the columns axis, you would just use normal bracket notation df[]. Calling one level of the index returns the sub-dataframe:

# In[ ]:


df.loc['G1']


# In[ ]:


df.loc['G1'].loc[1]


# In[ ]:


df.index.names


# In[ ]:


df.index.names = ['Group','Num']
df


# In[ ]:


df.xs('G1')


# In[ ]:


df.xs(['G1',1])


# In[ ]:


df.xs(1,level='Num')


# **Missing Data**
# 
# Let's show a few convenient methods to deal with Missing Data in pandas

# In[ ]:


# Again I import for you (however it is not requried)
import numpy as np
import pandas as pd


# In[ ]:


df = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}  # // we make dictonary
df 


# In[ ]:


d = pd.DataFrame(df)     # now we make a data frame with using pandas
d


# In[ ]:


d.dropna()   # it will drop all the NaN values from the table


# **Drop Na parameters**
# 
# Signature :
# df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# 
# Docstring :
# Return object with labels on given axis omitted where alternately any
# or all of the data are missing
# 
# Parameters
# ----------
# axis : {0 or 'index', 1 or 'columns'}, or tuple/list thereof
#     Pass tuple or list to drop on multiple axes
# how : {'any', 'all'}
#     * any : if any NA values are present, drop that label
#     * all : if all values are NA, drop that label
# thresh : int, default None
#     int value : require that many non-NA values
# subset : array-like
#     Labels along other axis to consider, e.g. if you are dropping rows
#     these would be a list of columns to include
# inplace : boolean, default False
#     If True, do operation inplace and return None.
# 
# Returns
# -------
# dropped : DataFrame
# 

# In[ ]:


d.dropna(axis=1)  # for column


# In[ ]:


d.dropna(thresh=2) # thresh = 2 means keep only the rows with atleast 2 Non-Na values 


# In[ ]:


d    # Note : I don't pass inplace = True that is why not rows or columns deleted permanently.


# In[ ]:


d.dropna(thresh = 3 )  # it means keep only those rows which having atleast 3 Non-Na values that is index 0


# In[ ]:


d.fillna(value='FILL VALUE')


# In[ ]:


d['A'].fillna(value=d['A'].mean())  # here we replace NaN value with mean of the column


# **Group By**
# 
# The groupby method allows you to group rows of data together and call aggregate functions

# In[ ]:


import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}


# In[ ]:


df = pd.DataFrame(data)
df


# ** Now you can use the .groupby() method to group rows together based off of a column name. For instance let's group based off of Company. This will create a DataFrameGroupBy object:**

# In[ ]:


df.groupby('Company')


# In[ ]:


# You can save this object as a new variable:
by_comp = df.groupby("Company")
by_comp


# In[ ]:


# And then call aggregate methods off the object:
by_comp.mean()


# In[ ]:


# or Simply write this
df.groupby('Company').mean()   # you can calculate other things like sum, standard deviation etc.


# In[ ]:


df.groupby('Company').std() # standard deviation


# In[ ]:


df.groupby('Company').sum()


# In[ ]:


df.groupby('Company').min()


# In[ ]:


by_comp.count()
# or 
df.groupby('Company').count()


# In[ ]:


by_comp.describe()
# or
# df.groupby('Company').describe()


# In[ ]:


by_comp.describe().transpose()


# In[ ]:


by_comp.describe().transpose()['GOOG']


# **Merging, Joining, and Concatenating**
# 
# There are 3 main ways of combining DataFrames together: Merging, Joining and Concatenating. In this lecture we will discuss these 3 methods with examples.

# In[ ]:


import pandas as pd


# In[ ]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])


# In[ ]:


df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 


# In[ ]:


df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])


# In[ ]:


df1


# In[ ]:


df2


# In[ ]:


df3


# ## Concatenation
# 
# Concatenation basically glues together DataFrames. Keep in mind that dimensions should match along the axis you are concatenating on. You can use **pd.concat** and pass in a list of DataFrames to concatenate together:

# In[ ]:


pd.concat([df1,df2,df3])


# In[ ]:


pd.concat([df1,df2,df3],axis=1)


# In[ ]:


left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})    


# In[ ]:


left


# In[ ]:


right


# ## Merging
# 
# The **merge** function allows you to merge DataFrames together using a similar logic as merging SQL Tables together. For example:

# In[ ]:


pd.merge(left,right,how='inner',on='key')


# In[ ]:


# Or to show a more complicated example:
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})


# In[ ]:


pd.merge(left, right, on=['key1', 'key2'])


# In[ ]:


pd.merge(left, right, how='outer', on=['key1', 'key2'])


# In[ ]:


pd.merge(left, right, how='right', on=['key1', 'key2'])


# In[ ]:


pd.merge(left, right, how='left', on=['key1', 'key2'])


# ## Joining
# 
# Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame.

# In[ ]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])


# In[ ]:


left.join(right)


# In[ ]:


left.join(right, how='outer')


# **Operations**
# 
# There are lots of operations with pandas that will be really useful to you, but don't fall into any distinct category. Let's show them here in this lecture:

# In[ ]:


import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()   # when you use head function it will display only 5 index by default
# however in this we have only 4 index/rows so it display all four index


# ### Info on Unique Values

# In[ ]:


df['col2'].unique()


# In[ ]:


df['col2'].nunique()
# Return Series with number of distinct observations over requested


# In[ ]:


df['col2'].value_counts()


# ### Selecting Data

# In[ ]:


#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]


# In[ ]:


newdf


# In[ ]:


### Applying Functions
def times2(x):
    return x*2


# In[ ]:


df['col1'].apply(times2)


# In[ ]:


df['col3'].apply(len)


# In[ ]:


df['col1'].sum()


# ** Permanently Removing a Column**

# In[ ]:


del df['col1']


# In[ ]:


df


# ** Get column and index names: **

# In[ ]:


df.columns


# In[ ]:


df.index


# ** Sorting and Ordering a DataFrame:**

# In[ ]:


df


# In[ ]:


df.sort_values(by='col2') #inplace=False by default


# ** Find Null Values or Check for Null Values**

# In[ ]:


df.isnull()


# In[ ]:


# Drop rows with NaN Values
df.dropna()


# ** Filling in NaN values with something else: **

# In[ ]:


import numpy as np


# In[ ]:


df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()


# In[ ]:


df.fillna('FILL')


# In[ ]:


data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)


# In[ ]:


df


# In[ ]:


df.pivot_table(values='D',index=['A', 'B'],columns=['C'])


# **Data Input and Output**
# 
# This notebook is the reference code for getting input and output, pandas can read a variety of file types using its pd.read_ methods. Let's take a look at the most common data types:

# In[ ]:


import numpy as np
import pandas as pd


# **CSV**
# 
# **CSV Input**

# In[ ]:


#df = pd.read_csv('../input/salaries/Salaries.csv')
#df


# In[ ]:


df.head()


# ## Excel
# 
# Pandas can read and write excel files, keep in mind, this only imports data. Not formulas or images, having images or macros may cause this read_excel method to crash. 

# In[ ]:


# Excel Input
pd.read_excel('../input/excel-file/Excel_Sample.xlsx',sheetname='Sheet1')


# **HTML**
# 
# You may need to install htmllib5,lxml, and BeautifulSoup4. In your terminal/command prompt run:
# 
# conda install lxml
# conda install html5lib
# conda install BeautifulSoup4
# Then restart Jupyter Notebook. (or use pip install if you aren't using the Anaconda Distribution)
# 
# Pandas can read table tabs off of html. For example:
# 
# **HTML Input**
# 
# Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects:

# In[ ]:


# df = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html')
# df

