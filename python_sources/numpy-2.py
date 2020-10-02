#!/usr/bin/env python
# coding: utf-8

# #                                                    NUMPY-2

# Python for Data Science- Module1

# ### Reshaping an Array

# Reshaping array using reshape() 

# In[ ]:


import numpy as np
x=np.arange(1,10,dtype=int)
print(x)


# In[ ]:


import numpy as np
x=np.arange(1,10,dtype=int).reshape(3,3)
print(x)


# Creating an array: for 2-D array use two square brackest [[a],[b]] ,and similarly for 3-D array three square bracket [[[a],[b],[c]]] 

# In[ ]:


a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)


# In[ ]:


a.shape #returns the dimensions of the array 


# --------------------------------------------------------------------------------------------------------------------------------
# ### Applying Operations(+, -, /, *) on Array

# In[ ]:


array1=np.array([[1,2,3],[4,5,6],[7,8,9]])
array2=np.arange(11,20).reshape(3,3)


# In[ ]:


array1


# In[ ]:


array2


# In[ ]:


np.add(array1,array2) #adding array1 and array2


# In[ ]:


np.multiply(array1,array2) #multiplying array1 and array2


# In[ ]:


np.divide(array1,array2) #dividing array1 and array2


# In[ ]:


np.remainder(array1,array2) #remainder of array1 and array2


# In[ ]:


np.subtract(array1,array2) #subtracting array1 and array2


# --------------------------------------------------------------------------------------------------------------------------------
# ### Array Slicing  

# array_name[row,column], and to acess particular row or column apply index slicing 

# In[ ]:


a[:2,:] 


# In[ ]:


a[1:3,2:3]


# --------------------------------------------------------------------------------------------------------------------------------
# ### Applying Some Matrix Function 

# - numpy.transpose(array)
# - numpy.append(array, axis)
# - numpy.insert(input_array,index_position,values_tobe_inserted,axis)
# - numpy.delete(input_array,index_position_tobe_deleted,axis)
# 
# for row axis=0,and axis=1 for column
# 

# In[ ]:


np.transpose(a)


# In[ ]:


x=np.arange(2,5).reshape(3,1)
print(x)


# In[ ]:


np.append(a,x,axis=1)


# In[ ]:


x=np.arange(2,5).reshape(1,3)
np.append(a,x,axis=0)


# In[ ]:


x=np.insert(a,2,[0,0,0],axis=0)
print(x)


# In[ ]:


np.delete(x,2,axis=0)

