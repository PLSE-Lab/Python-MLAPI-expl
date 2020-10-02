#!/usr/bin/env python
# coding: utf-8

# # NumPy
# 
# 
# NumPy is the fundamental package for scientific computing with Python. It contains among other things:
# 
# - a powerful N-dimensional array object <br>
# - sophisticated (broadcasting) functions <br>
# - tools for integrating C/C++ and Fortran code <br>
# - useful linear algebra, Fourier transform, and random number capabilities

# In[1]:


# Libraries
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Python list allows different data types
p_list = [False, "Text", 7.5, 42, np]

[type(item) for item in p_list]


# ## 1- Numpy Dtypes

# In[3]:


# Numpy array allows only data type
np_list = np.array([100, 5, 6.7, 8, 5] , dtype=int)

print(type(np_list))
[type(item) for item in np_list]


# 
# ## 2 - Tree important attributes for a NumPy Array
# 

# In[4]:


print(np_list.dtype)
print(np_list.shape)
print(np_list.ndim)
print(len(np_list))


# ## 3 - Generating Numpy Arrays

# In[5]:


np1 = np.zeros((2,4))
np1


# In[6]:


np2 = np.ones((4,6), dtype='int16')    
np2


# In[7]:


np3 = np.eye(3)
np3


# ***np.arange(n)*** is the equivalent Numpy function for ***range(n)***

# In[8]:


# Creating a Numpy Array
print('np.arange( 10) = ', np.arange(10))
print('np.arange( 3, 8) = ', np.arange(3,8))
print('np.arange( 0, 2, 0.5) = ', np.arange(0, 2, 0.5))
print('np.linspace( 0, 2, 5 ) = ', np.linspace(0, 10, 5))


# ## 4 - Slicing Numpy Arrays
# 
# ### slice = ndarray[start:stop:step]

# In[9]:


# vector[start:stop:step]
a = np.arange(20)
print(a[1:17:2])
print()


# In[10]:


# Negative indexes
list_a = np.arange(20)
print('Negative indexes counts backwords from last position')
print(a[0:-1])
print()


# ###  Slicing Multidimensional Arrays

# In[11]:


# nested lists result in multi-dimensional arrays, using List Comprehensions
np_vet = np.array([range(i, i + 4) for i in [10, 20, 30]])
np_vet


# In[12]:


np_vet[:,:]


# ## 5 - Saving and Loding Numpy Arrays

# In[15]:


np.savez_compressed('arq' , v1=np_vet, v2=list_a)


# In[16]:


load_vet = np.load('arq.npz')
print(load_vet['v1'], '\n')
print(load_vet['v2'], '\n')


# ## 6 - Reshaping Numpy objects

# In[17]:


# Only can reshape compatible number of elements
grid = np.arange(1, 22).reshape((3, 7 ))
print(grid)


# ## 7 - Concatenating and Splitting with Numpy

# In[18]:


x = np.array([1, 2, 3, 5 , 6])
y = np.array([3, 2, 1])
np.concatenate([x, y])


# In[19]:


grid = np.array([[1, 2, 3],
                 [4, 5, 6]])


# In[20]:


# concatenate along the first axis
np.concatenate([grid, grid])


# In[21]:


# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)


# In[22]:


x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])


# In[23]:


# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])


# In[24]:


x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)


# ## 8 - Numpy Functions and Computations - Doing the magic! 

# In[25]:


tempC = np.random.normal(25, 2.5, 10000000)
tempC[0:50]
#len(tempC)


# In[26]:


get_ipython().run_cell_magic('time', '', 'for temp in tempC:\n    temp = (temp * 1.8) + 32 # Celsius to Fahrenheit')


# In[27]:


get_ipython().run_cell_magic('time', '', 'tempF = (tempC * 1.8) + 32\n#tempF =  np.add(np.multiply(tempC,1.8), 32) ')


# In[28]:


tempF[:20]


# In[29]:


get_ipython().run_cell_magic('time', '', '#average tempeture\nnp.mean(tempC)')


# In[30]:


get_ipython().run_cell_magic('time', '', "print('sum: ',tempC.sum())\nprint('max: ',tempC.max())\nprint('min: ',tempC.min())\nprint('mean:',tempC.mean())\nprint('median:',np.median(tempC))\nprint('std: ',tempC.std())\nprint('var:',np.var(tempC), '\\n')")


# In[31]:


get_ipython().run_cell_magic('time', '', 'np.add.reduce(tempC)')


# ### Exponential and Logarithmic Functions
# 

# In[47]:


x = [1, 2, 3, 4]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))


# In[36]:


x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

#import math
#print('\n', math.log(2**4, 2))


# ### Applying Functions by Axis

# In[37]:


M = np.random.random((3, 4))
print(M)


# In[38]:


M.min(axis=0)


# In[39]:


M.max(axis=1)


# ## 9 - Converting Numpy Arrays to Pandas Dataframes

# In[40]:


import pandas as pd
df = pd.DataFrame(grid)
df


# In[41]:


np.array(df)


# ## 10 - Using all this!

# In[48]:


from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

# Read mminist dataset - this is a very popular dataset with hand written digits from 0 to 9
train = np.loadtxt("../input/train.csv", delimiter=',', skiprows=1)


# In[49]:


#  42000 numbers with 32 x 32 pixels,
#  first coluns has the value of the number
print(train.shape)
train[0:1,:]


# In[50]:


#Now we separate the target class (y) from the features x vector
X_train = train[:, 1:] # gets all rows, all columns except the first one (index 0) 
X_train[0:5, :] # print first 5 rows


# In[51]:


y_train = train[:, 0:1] # gets all rows, only first column (index 0) 
y_train[0:5, :] # print first 5 rows


# In[52]:


# for a simple test let us print the first number of the train list
plt.imshow(np.reshape(X_train[0],(28,28)),cmap = 'gray') 
print('class:',y_train[0])


# ### Creating Noise Filter from Scratch

# In[62]:


for n, digit in enumerate(X_train[:4]):
#for digit in X_train[:4]:
    filtered = np.reshape(digit,(28,28)) + np.random.normal(127, 20.0, (28,28))
    plt.imshow(filtered,cmap = 'gray')
    plt.xlabel(y_train[n])
    plt.show()


# ### Recap Challenge

# How can you convert the digits to a forecoler Black and Background White image?
