#!/usr/bin/env python
# coding: utf-8

# # 05 Arrays

# - Creating and Manipulating 1D & 2D Arrays
# - Array Operations

# ## 1D Numpy Array

# ## Create array

# In[ ]:


import numpy as np  


# In[ ]:


a = np.array([0,1,2,3,4])


# In[ ]:


print(f'Numpy Array a \n{a}')
print(f'Type of Numpy Array a {type(a)}')
print(f'Elements Type of Numpy Array a {a.dtype}')
print(f'Size of Numpy Array a {a.size}')
print(f'Dimensions of Numpy Array a {a.ndim}')
print(f'Shape of Numpy Array a {a.shape}')


# ## Manipulate Array

# ## Indexing

# In[ ]:


a[0]


# In[ ]:


for index,element in enumerate(a):
    print(f'index {index} element {element}')


# In[ ]:


b=np.array([3.1,11.02,6.2,231.2,5.2])


# In[ ]:


print(f'Numpy Array b {b}')
print(f'Type of Numpy Array b {type(b)}')
print(f'Elements Type of Numpy Array b {b.dtype}')
print(f'Size of Numpy Array b {b.size}')
print(f'Dimensions of Numpy Array b {b.ndim}')
print(f'Shape of Numpy Array b {b.shape}')


# In[ ]:


for index,element in enumerate(b):
    print(f'index {index} element {element}')


# In[ ]:


c=np.array([20,1,2,3,4])
c


# In[ ]:


c[0]=100
c


# In[ ]:


c[4]=0
c


# ## Slice

# In[ ]:


d = c[1:4]
d


# In[ ]:


d.size


# In[ ]:


c[3:5]=300,400
c


# ## Array Operations

# ## Vector Addition & Subtraction

# In[ ]:


u = np.array([1,0])
u


# In[ ]:


v = np.array([0,1])
v


# In[ ]:


z = u + v
z


# In[ ]:


type(z)


# In[ ]:


z = u-v
z


# ## Array multiplication with a Scalar

# In[ ]:


y = np.array([1,2])
y


# In[ ]:


z = 2*y
z


# ## Product of two Numpy arrays

# In[ ]:


u = np.array([1,2])
u


# In[ ]:


v = np.array([3,1])
v             


# ### Hadamard Product

# In[ ]:


z = u*v
z


# ### Dot Product

# In[ ]:


u.T


# In[ ]:


z= np.dot(u,v)
z


# In[ ]:


z = u@v
z


# ## Broadcasting

# ### Adding Constant to an Numpy Array

# In[ ]:


u = np.array([1,2,3,-1])
u


# In[ ]:


z = u + 1
z


# ## Universal Functions

# In[ ]:


a = np.array([1,-1,1,-1])
a


# In[ ]:


mean_a = a.mean()
mean_a


# In[ ]:


b = np.array([1,-2,3,4,5])
b


# In[ ]:


max_b = b.max()
max_b


# In[ ]:


np.pi


# In[ ]:


x = np.array([0,np.pi/2,np.pi])
x


# In[ ]:


y = np.sin(x)
y


# In[ ]:


np.linspace(-2,2,num=5)


# In[ ]:


np.linspace(-2,2,num=9)


# ## Plotting 

# In[ ]:


x = np.linspace(0,2*np.pi,100)
x


# In[ ]:


y = np.sin(x)
y


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.plot(x,y);


# # 2D Numpy Array

# ## Create 2D Numpy Array

# In[ ]:


A = np.array([[11,12,13],
             [21,22,23],
             [31,32,33]])
print(f'Numpy Array A \n{A}')
print(f'Type of Numpy Array A {type(A)}')
print(f'Elements Type of Numpy Array A {A.dtype}')
print(f'Size of Numpy Array A {A.size}')
print(f'Dimensions of Numpy Array A {A.ndim}')
print(f'Shape of Numpy Array A {A.shape}')


# ### Indexing

# In[ ]:


A[0]


# In[ ]:


A[0][0]


# In[ ]:


A[1][2]


# ### Slicing

# In[ ]:


A[0,0:2]


# In[ ]:


A[1,:]


# In[ ]:


A[:,2]


# In[ ]:


A[0:2,2]


# ## Addition

# In[ ]:


X = np.array([[1,0],
              [0,1]])
X


# In[ ]:


Y = np.array([[2,1],
              [1,2]])
Y


# In[ ]:


Z = X + Y
Z


# In[ ]:


Z = 2*Y
Z


# # Matrix Multiplication

# ## Hadamard Matrix Multiplication

# In[ ]:


Z = X * Y
Z


# ## Dot Matrix Multiplication

# In[ ]:


Z = X@Y
Z


# In[ ]:


A = np.array([[0,1,1],
              [1,0,1]])
A


# In[ ]:


B = np.array([[1,1],
              [1,1],
              [-1,1]])
B


# In[ ]:


C = A@B
C


# # Plotting functions

# In[ ]:


# Plotting functions Plotvec1,Plotvec2

def Plotvec1(u, z, v):
    
    ax = plt.axes()
    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(u + 0.1), 'u')
    
    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(v + 0.1), 'v')
    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

def Plotvec2(a,b):
    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05, color ='r', head_length=0.1)
    plt.text(*(a + 0.1), 'a')
    ax.arrow(0, 0, *b, head_width=0.05, color ='b', head_length=0.1)
    plt.text(*(b + 0.1), 'b')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)


# In[ ]:


u = np.array([1, 0])
v = np.array([0, 1])
z = u + v


# In[ ]:


Plotvec1(u, z, v)
print(f"The dot product u@z is {u@z}")


# In[ ]:


a,b = np.array([-1,1]),np.array([1,1])
Plotvec2(a,b)
print(f"The dot product a@b is {a@b}")


# In[ ]:


a,b = np.array([1,0]),np.array([0,1])
Plotvec2(a,b)
print(f"The dot product a@b is {a@b}")


# In[ ]:


# The vectors are perpendicular. 
# As a result, the dot product is zero. 

