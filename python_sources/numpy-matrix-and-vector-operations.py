#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# ### Basic operations using numpy and pandas 
# ### using numpy we can do matrix and vector releated operations
# ### using pandas we can do data manipulations like load data , analyze data , clean data 

# In[ ]:


#create array using numpy 
arr=np.array([1,2,3,4,5]) # one dimensional array 
arr1=np.array([[1,2,3],[4,5,6,],[7,8,9]]) # two dimensional array 
print(arr)
print(arr1)


# In[ ]:


arrzero=np.zeros([5,5]) # 5*5 Matrix
arrone=np.ones([5,5])
arrempty=np.empty([5,5])
print(arrzero) 
print(arrone)
print(arrempty)


# In[ ]:


# slice operations

print(arr1) # if we want to retrive specific item we can use slice concept 

print(arr1[:]) # to retrive all elements from matrix 
print(arr1[1:2]) # Row1 and column 2 
print(arr1[:,:1]) # all rows butonly first columns
print(arr1[:2,:]) ## only row rows all columns


# In[ ]:


# Now just creating vector 
vector1=[1,2,3,4]
print(vector1)


# In[ ]:


# To calculate vector lenght we can do 3 ways 
# 1. Euclidean distance (or) first normal form
# 2. Manhatten distance (or) second normal form
# 3. Max normal form 


# In[ ]:


from scipy.linalg import norm
from math import inf


# In[ ]:


print(norm(vector1,1)) # Manhatten distance


# In[ ]:


print(norm(vector1,inf)) # Max normal form 


# In[ ]:


print(norm(vector1,2)) # Euclidean distance


# In[ ]:


# Type of matrix 
# 1. Square matrix 
# 2. Triangular matrix (Lower triangular and upper triangular )
# 3. Identity matrix 
# 4. symmetric matrix 
# 5. orthogonal matrix
# 6. diagonal matrix 


# In[ ]:


sqarematrix =np.array([[1,2,3],[42,55,63],[7,8,9]])
print(sqarematrix)
sqarematrix.shape # matrix size rows and columns 


# In[ ]:


print(np.identity(3)) # create Identity matrix 


# In[ ]:


np.diag(sqarematrix) # using diag method we can retrive main diagonal values 


# In[ ]:


np.tril(sqarematrix) # lower triangular matrix 


# In[ ]:


np.triu(sqarematrix) # upper triangular matrix 


# In[ ]:


# matrix operations 
# 1. Inverse 
# 2. Transpose
# 3. rank 
# 4. determents 
# 5. sudo inverse for rectangular matrix


# In[ ]:


from numpy.linalg import inv,pinv,matrix_rank,det


# In[ ]:


print(inv(sqarematrix))


# In[ ]:


#print(pinv(sqarematrix))


# In[ ]:


print(matrix_rank(sqarematrix))


# In[ ]:


print(det(sqarematrix))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




