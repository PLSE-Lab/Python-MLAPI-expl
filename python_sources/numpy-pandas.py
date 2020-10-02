#!/usr/bin/env python
# coding: utf-8

# ## Playing with Numpy & Pandas

# Importing the numpy and Pandas into the notebook

# In[ ]:


import numpy as np
import pandas as pd


# Assiging an array to a variable using np

# In[ ]:


a=np.array([34,43,43,434,2,4,4])


# Finding the mean,size,shape and dimensions of the variable assigned using the attributes of numpy. 

# In[ ]:


print('mean of a:',a.mean())
print('size of a:',a.size)
print('shape of a:',a.shape )
print('dinmension of a:',a.ndim)


# Can also view the vairable as different datatype 

# In[ ]:


a.astype(float)


# Can assign a random list(array) of items to a variable using numpy 

# In[ ]:


b=np.arange(1,21)


# In[ ]:


b


# Finding the size,shape and dimensions of the variable created using the attributes of numpy

# In[ ]:


print('Shape:',b.shape)
print('Size:',b.size)
print('dimension:',b.ndim)


# Using the reshape attribute to change the shape of the variable

# In[ ]:


b1=b.reshape(5,4) #5 rows and 4 columns
b2=b.reshape(4,5) #4 rows and 5 columns

print("b1:\n",b1)
print("\nb2:\n",b2)


# #### Scalar operations

# In[ ]:


b1*10


# In[ ]:


b1+10


# Multiplication is possible as b1 has 5*4 and b2 has 4*5.. Thus resulting a 5*5 matrix

# In[ ]:


b1.dot(b2)


# Multiplication is not possible as b1 has 5*4 and a has 1 dimension

# In[ ]:


b1.dot(a)


# Transpose is possible

# In[ ]:


print('b1:\n',b1)
print('Transpose:\n',b1.transpose())


# Create a 2*2 matrix using numpy array

# In[ ]:


c=np.array([[1.,2.],[3.,4.]])


# In[ ]:


c


# Inverse of an array/matrix can be found using this attribute

# In[ ]:


np.linalg.inv(c)


# creating a 2*2 array and reshaping it for further operations

# In[ ]:


d=np.array(np.arange(9).reshape(3,3),dtype=float)


# In[ ]:


d


# In[ ]:


#print 1.,4.,7.
#d[:,-2]
d[:,1]


# In[ ]:


#print 1,2,4,5
d[:2,1:]


# In[ ]:


#print 1,2,4,5,7,8
#d[:,1:]
d[:,-2:]


# In[ ]:


#print 3,4,6,7
d[1:3,0:2]


# In[ ]:


d[d>5]


# create an 1d array by simmply assiging a list of items to a variable

# In[ ]:


e=[23,3,4,34,34,3,43,43,4]


# In[ ]:


e[6:]


# In[ ]:


e[-3:]


# create an array of zeroes and ones from numpy

# In[ ]:


np.zeros(10)


# In[ ]:


np.ones(10)


# Create 5*5 matirix with 5 using numpy.ones and numpy.zeros

# In[ ]:


(np.ones(25)*5).reshape(5,5)


# In[ ]:


np.ones(25).reshape(5,5)*5


# In[ ]:


np.zeros(25).reshape(5,5)+5


# #### Constructing a Dataframe from a Dictionary

# In[ ]:


Name_Age={'name':['Raghav','Sharmi','Ishaan'],'Age':[28,28,1]}


# In[ ]:


type(Name_Age)


# In[ ]:


family=pd.DataFrame(Name_Age)


# In[ ]:


family
#By default, row labels(Index) are from the range(0-3) as we have 3 records


# In[ ]:


type(family)


# #### Broadcasting with a dictionary

# In[ ]:


new_data=np.arange(150)

new_col={'new_column':new_data,'sex':'M'}

new_column=pd.DataFrame(new_col)


# In[ ]:


new_column.head(5)


# In[ ]:


new_column.tail(5)


# In[ ]:


#change the column index and Column name

new_column.index=np.arange(150,300)


# In[ ]:


new_column.columns=['numbers','gender']


# In[ ]:


new_column.head(5)


# #### Combining and Merging DataSets

# Create multiple Dataframes for different datasets
# 
# 
# 
# Use concat/Merge to join the datasets. 

# In[ ]:


test_dataA={'Id':[100,101,102,103,104],'First_Name':['Raghav','Sankar','Akhil','Deepa','Sufiya'],'Last_Name':['v','Jayaram','Vijayan','Nayak','Taranam']}


# In[ ]:


df1=pd.DataFrame(test_dataA,columns=['Id','First_Name','Last_Name'])


# In[ ]:


test_dataB={'Id':[105,106,107,108,109],'First_Name':['Renju','Saurabh','Pradeep','Joshua','Mythreyi'],'Last_Name':['Nair','Indulkar','Bhat','Singh','Matada']}


# In[ ]:


df2=pd.DataFrame(test_dataB,columns=['Id','First_Name','Last_Name'])


# In[ ]:


df1


# In[ ]:


df2


# In[ ]:


df_join=pd.concat([df1,df2])


# In[ ]:


df_join


# In[ ]:


testdataC={'Id':[100,101,102,103,104,105,106,107,108,109],'Expense_Id':[100,200,260,500,490,600,390,290,340,111]}


# In[ ]:


df3=pd.DataFrame(testdataC,columns=['Id','Expense_Id'])


# In[ ]:


df3


# In[ ]:


pd.merge(df_join,df3,on='Id')


# #### String Manipulation

# In[ ]:


string='India is my country'


# In[ ]:


#capitalize() helps to change the first letter of the string to capitalize

uppercase= string.capitalize()


# In[ ]:


uppercase


# In[ ]:


#islower() helps to check whether the full string has lower case or not.. Returns TRUE or FALSE 

string.islower()


# In[ ]:


#isupper() helps to check whether the full string has upper case or not.. Returns TRUE or FALSE 

string.isupper()


# In[ ]:


#len() helps to find the count of the letters available in the string.. returns 'int' datatype

len(string)


# In[ ]:


#lower() helps to lower all the letters available in the string.
#upper() helps to upper all the letters available in the string.

string.lower()


# In[ ]:


#title() helps to capitalise the first letter of all the words in a string. 

string.title()


# In[ ]:


#swapcase() helps to convert all the letters available in the sring to convert to reversecase

string.swapcase()


# ### We shall discuss more on the other topics later.. Keep following... 
# 

# ### Thanks.. 

# ### Happieeeee Learning... 
