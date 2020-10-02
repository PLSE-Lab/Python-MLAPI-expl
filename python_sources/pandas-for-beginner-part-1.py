#!/usr/bin/env python
# coding: utf-8

# The purpose of making this Kernel is to get the begineers aquinted with Pandas module.Pandas is very import for data preprocessing and manuplation.This kernel is work in process and I will be updating this in the coming days.If you like my work please do vote.

# **A] Pandas Series**

# In[ ]:


import numpy as np 
import pandas as pd 
from pandas import Series,DataFrame


# In[ ]:


obj=Series([3,6,9,12])
obj


# In[ ]:


obj.values


# In[ ]:


obj.index


# In[ ]:


# Creating series 
ww2_cas=Series([87,43,30,21,40],index=['USSR','Germany','China','Japan','USA'])
ww2_cas


# In[ ]:


'USSR' in ww2_cas


# In[ ]:


# Putting the Series into a Dictionary 
ww2_dict=ww2_cas.to_dict()
ww2_dict


# In[ ]:


# Converting a dictionary to Series
ww2_series=Series(ww2_dict)
ww2_series


# In[ ]:


countries=['China','Germany','Japan','USA','USSR','Argentina']


# In[ ]:


obj2=Series(ww2_dict,index=countries)
obj2


# In[ ]:


pd.isnull(obj2)


# In[ ]:


pd.notnull(obj2)


# In[ ]:


# Adding two series
ww2_series+obj2


# In[ ]:


# Naming a Series or Object
obj2.name='World War II Casualties'
obj2


# In[ ]:


# Giving Name to the Index4
obj2.index.name='Countries'
obj2


# **B] Index Objects**

# In[ ]:


import numpy as np 
from pandas import Series,DataFrame
import pandas as pd


# In[ ]:


my_ser=Series([1,2,3,4],index=['A','B','C','D'])
my_ser


# In[ ]:


my_index=my_ser.index
my_index


# In[ ]:


my_index[2]


# In[ ]:


my_index[2:]


# In[ ]:


my_index[0]


# In[ ]:


#Index are immutable, we will get an error if we try to do this
#my_index[0]='Z'


# **C] Reindex**

# In[ ]:


import numpy as np 
from pandas import Series,DataFrame
import pandas as pd

from numpy.random import randn


# In[ ]:


# Creating a Series with Indices A,B,C,D and values 1,2,3,4
ser1=Series([1,2,3,4],index=['A','B','C','D'])
ser1


# In[ ]:


# New indexes E and F are added to the serie with default NaN Values 
ser2=ser1.reindex(['A','B','C','D','E','F'])
ser2


# In[ ]:


#Here the newly added index G is filled with Zero Value
ser2a=ser2.reindex(['A','B','C','D','E','F','G'],fill_value=0) 
ser2a


# In[ ]:


ser3=Series(['USA','Mexico','Canada'],index=[0,5,10])
ser3


# In[ ]:


# Range command to make list of numbers
ranger=range(15)
ranger


# In[ ]:


# Using forward fill (ffill) command to create a series 
ser3.reindex(ranger,method='ffill')


# In[ ]:


# Making a dataframe using a randn function
dframe=DataFrame(randn(25).reshape((5,5)),index=['A','B','D','E','F'],columns=['col1','col2','col3','col4','col5'])
dframe


# In[ ]:


# Reindexing rows of a dataframe
dframe2=dframe.reindex(['A','B','C','D','E','F'])
dframe2


# In[ ]:


# Reindexing columns of a dataframe
new_columns=['col1','col2','col3','col4','col5','col6']
dframe2a=dframe2.reindex(columns=new_columns)
dframe2a


# In[ ]:


# Reindexing rows and columns using ix command 
dframe
# C and col6 will be added to the daframe using ix command col1


# In[ ]:


dframe_a=dframe.ix[['A','B','C','D','E','F'],new_columns]
dframe_a


# **D] Drop Entry**

# In[ ]:


import numpy as np
from pandas import Series,DataFrame
import pandas as pd 


# In[ ]:


ser1=Series(np.arange(3),index=['a','b','c'])
ser1


# In[ ]:


ser1.drop('b')


# In[ ]:


dframe1=DataFrame(np.arange(9).reshape((3,3)),index=['SF','LA','NY'],columns=['pop','size','year'])

dframe1


# In[ ]:


# Droppping row
dframe2=dframe1.drop('LA')
dframe2


# In[ ]:


# Dropping columns
dframe1.drop('year',axis=1)


# **E] Selecting entries **

# In[ ]:


import numpy as np
import pandas as pd 
from pandas import Series,DataFrame


# In[ ]:


ser1=Series(np.arange(3),index=['A','B','C'])
ser1=2*ser1
ser1


# In[ ]:


ser1['B']


# In[ ]:


ser1[1]


# In[ ]:


ser1[0:3]


# In[ ]:


ser1[['A','B']]


# In[ ]:


# Displaying values greater than 3
ser1[ser1>3]


# In[ ]:


# Replacing values
ser1[ser1>3]=10
ser1


# In[ ]:


dframe=DataFrame(np.arange(25).reshape((5,5)),index=['NYC','LA','SF','DC','Chi'],columns=['A','B','C','D','E'])
dframe


# In[ ]:


dframe['B']


# In[ ]:


dframe[['B','E']]


# In[ ]:


dframe[dframe['C']>8]


# In[ ]:


# Getting info in Boolean format
dframe>10


# In[ ]:


dframe.ix['LA']


# In[ ]:


dframe.ix[1]


# **F] Data Alignment**

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[ ]:


ser1=Series([0,1,2],index=['A','B','C'])
ser1


# In[ ]:


ser2=Series([3,4,5,6],index=['A','B','C','D'])
ser2


# In[ ]:


ser1+ser2


# In[ ]:


dframe1=DataFrame(np.arange(4).reshape((2,2)),columns=list('AB'),index=['NYC','LA'])
dframe1


# In[ ]:


dframe2=DataFrame(np.arange(9).reshape((3,3)),columns=list('ADC'),index=['NYC','SF','LA'])
dframe2


# In[ ]:


dframe1 + dframe2


# In[ ]:


dframe1


# In[ ]:


dframe1.add(dframe2,fill_value=0)


# In[ ]:


ser3=dframe2.ix[0]
ser3


# In[ ]:


dframe2-ser3


# **G] Sorting and Ranking**

# In[ ]:


import numpy as np
from pandas import Series,DataFrame
import pandas as pd


# In[ ]:


ser1=Series(range(3),index=['C','A','B'])
ser1


# In[ ]:


ser1.sort_index()


# In[ ]:


from numpy.random import randn


# In[ ]:


ser2=Series(randn(10))
ser2


# In[ ]:


ser1.sort_values()


# In[ ]:


ser2.sort_values()


# In[ ]:


ser2.rank()


# **H] Summary of Stats**

# In[ ]:


arr=np.array([[1,2,np.nan],[np.nan,3,4]])
arr


# In[ ]:


dframe1=DataFrame(arr,index=['A','B'],columns=['One','Two','Three'])
dframe1


# In[ ]:


# Sum of colums 
dframe1.sum()


# In[ ]:


# Sum of rows
dframe1.sum(axis=1)


# In[ ]:


dframe1.min()


# In[ ]:


# Getting the index whoes value is minmum
dframe1.idxmin()


# In[ ]:


# Cumulative Sum
dframe1.cumsum()


# In[ ]:


dframe1


# In[ ]:


dframe1.describe()


# In[ ]:


# importing You tube videos
from IPython.display import YouTubeVideo


# In[ ]:


YouTubeVideo('xGbpuFNR1ME')


# In[ ]:


YouTubeVideo('4EXNedimDMs')


# In[ ]:


# Getting information from Web
import pandas_datareader.data as pdweb
import datetime


# In[ ]:


prices=pdweb.get_data_yahoo(['CVX','XOM','BP'],start=datetime.datetime(2010,1,1),end=datetime.datetime(2013,1,1))['Adj Close']
prices.head()


# In[ ]:


volume=pdweb.get_data_yahoo(['CVX','XOM','BP'],start=datetime.datetime(2010,1,1),end=datetime.datetime(2013,1,1))['Volume']
volume.head()


# In[ ]:


rets=prices.pct_change()


# In[ ]:


#Correlation of stocks
corr=rets.corr()


# In[ ]:


prices.plot()


# In[ ]:


import seaborn as sns
import matplotlib as plt


# In[ ]:


#sns.heatmap(rets,annot=False,diag_names=False)
ser1=Series(['w','w','x','y','z','w','x','y','x','a'])
ser1


# In[ ]:


# Getting the unique value 
ser1.unique()


# In[ ]:


# Finding out how many unique values 
ser1.value_counts()


# **I] Missing Value**

# In[ ]:


import numpy as np
from pandas import Series,DataFrame
import pandas as pd 


# In[ ]:


data=Series(['one','two',np.nan,'four'])
data


# In[ ]:


data.isnull()


# In[ ]:


# Dropping null values
data.dropna()


# In[ ]:


dframe=DataFrame([[1,2,3],[7,np.nan,9],[np.nan,np.nan,np.nan]])
dframe


# In[ ]:


clean_dframe=dframe.dropna()
clean_dframe


# In[ ]:


dframe.dropna(how='all')
#Drops the rows where all the values were NAN


# In[ ]:


# Dropping by columns
dframe.dropna(axis=1)


# In[ ]:


npn=np.nan
dframe2=DataFrame([[1,2,3,npn],[2,npn,5,6],[npn,7,npn,9],[1,npn,npn,npn]])
dframe2


# In[ ]:


dframe2.dropna(thresh=2)
# Rows with altleast two non nan values will be retained


# In[92]:


dframe2.dropna(thresh=3)


# In[93]:


dframe2


# In[95]:


# Fill NaN with 1
dframe.fillna(1)


# In[96]:


Fidframe2


# In[98]:


# Replacing null values with different values for column
dframe2.fillna({0:0,1:1,2:2,3:3})


# In[99]:


dframe2.fillna(0)


# In[100]:


dframe2


# In[101]:


# Overwrite the frame with new values 
dframe2.fillna(0,inplace=True)
dframe2


# In[ ]:




