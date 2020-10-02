#!/usr/bin/env python
# coding: utf-8

# # Intoduction To PyTorch 
# 
# ### Understanding torch.Tensor
# 
# PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab.
# 
# In this  notebook we will better understand one of the basic class of PyTorch, Tensor.
# 
# This Tensor class is present in the Module torch.
# 
# So in the next line we import the required Module which is torch.
# 
# In this notebook we will first briefly discuss various ways to create Tensors then we will disscus these 5 Interesting Funstions associated with the Class torch.Tensor.
# 

# In[ ]:


# Import torch and other required modules
import torch
import numpy as np


# ### **torch.Tensor**
# 
# Tensor class in PyTorch is a Multidimensional matrix containing Homogenous elements. These are the building blocks of any program we will write using PyTorch.
# 
# So let's create Objects(Tensor) of this class then we will see the 5 Interesting Functions of the Class.

# using torch.Tensor()  which is a constructor of the class.
# Inside the constructor we can pass any list, tuple, numpy array to crearte a Tensor with the Same Data.

# In[ ]:


# Example 1 - working (change this)
li=[
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,14,15],
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
]
torch.tensor(li)


# In[ ]:


tu=tuple(li)
torch.tensor(tu)


# In[ ]:


arr=np.array(tu)
torch.tensor(arr)


# But here the Dimensions must be uniform across the Tensor. Otherwise we cannot build a Tensor.

# In[ ]:


li1=[
    [
        [1,2],  # Here we have removed 3
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,15], #here we have removed 14
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
]
torch.tensor(li1)


# So here also we can see there is not uniformity but li3 can be executed.

# In[ ]:





# In[ ]:


li2=[
    [
        [1,2],  # Here we have removed 3
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11], # here we have removed 12
        [13,14,15], 
        [16,17,18]
    ],
    [
        [19,21], # Here we have removed 20
        [22,23,24],
        [25,26,27]
    ]
]
torch.tensor(li2)


# In[ ]:


li3=[
    [
        [1,2],  
        [5,6],
        [8,9]
    ],
    [
        [10,11], 
        [13,15], 
        [16,18]
    ],
    [
        [19,21], 
        [22,23],
        [25,26]
    ]
]
torch.tensor(li3)


# There are some other functions that are also used to create tensors some of them are as follows:

# **torch.arange(start,end,step)**
# 
# just like the arange() function in numpy, is used to create tensor with values which starts from "start"(default being 0) and ends at "end-1" with the default step being 1.

# In[ ]:


# A tensor with values 0 to 9
x=torch.arange(10)

# A tensor with multiples of 3 upto 30(including)
x1=torch.arange(3,31,3) # Since the last value to be considered is 31-1=30

x,x1


# **torch.randn(num)** 
# 
# It will return a tensors with "num" number of random values from a normal distribution(with mean=0 and variance=1 )
# 
# As we can see below we can also create N-Dimensional Tensor as well.

# In[ ]:


y=torch.randn(5)

# Creating a 2D Tensor
y1=torch.randn((3,3))

# Creating a 3D Tensor
y2=torch.randn((2,3,4))

y,y1,y2


# **torch.logspace(start,end,steps=100,base=10.0)**
# 
# Returns 1D tensor of "steps" number of values logarithmically spaced numbers with Base "base"  between $\text{base}^\text{start} $ and $\text{base}^\text{end} $

# In[ ]:


# 5 logarithmically spaced values between 2^1 and 2^10
z=torch.logspace(1,10,5,base=2)

# 10 logarithmically spaced values between 10^0 and 10^2
z1=torch.logspace(0,2,steps=10,base=10)

# 2 logarithmically spaced values netween 4^0 and 4^2
z2=torch.logspace(0,2,2,4)

z,z1,z2


# **torch.linspace(start, end, steps=100)** 
# 
# Returns 1D tensor of length "steps" equally spaced points between "start" and "end".

# In[ ]:


# 5 linearly spaced numbers from 1 to 10
a=torch.linspace(1,10,5)

# 5 linearly spaced numbers from 2 to 12
a1=torch.linspace(2,12,6)

a,a1


# To confirm that indeed tensor created by **torch.tensor()** is the same as the tensors created by above functions:

# In[ ]:


type(x),type(y),type(z),type(a), type(torch.tensor([]))


# ## Let's look at some of the interesting Functions 
#     1) torch.Tensor.register_hook(hook)
#     2) torch.Tensor.to_sparse(sparseDims)
#     3) torch.Tensor.dot(input, tensor)
#     4) torch.Tensor.flatten(input, start_dim=0, end_dim=-1) 
#     5) torch.Tensor.numel(input)

# # Function 1
# ## **torch.Tensor.register_hook(self, hook)**
# 
# When this function is executed we will call the registered hook whenever a gradient with respect to the Tensor is computed.
# 
# So say we have declared a tensor on which we can call backward() i.e. require_grad is set to True but say we want to apply a function on the gradient and store the result of that function in the the ".grad" variable then this function is used.
# 
# Also the function gives a handler which can be usedd to remove the hook by calling 
# 
# torch.Tensor.register_hook(self, hook).remove()

# In[ ]:


v = torch.tensor([0., 0., 0.], requires_grad=True)
v.backward(torch.tensor([0., 2., 3.]))
v.grad


# In[ ]:


# So here we will never get a Gradient which is zero the same can be extended to always positive gradients and 
# always greater than zero gradients
def f1(grad):
    for i in range(grad.numel()):
        if grad[i]==0:
            grad[i]+=1

v = torch.tensor([0., 0., 0.], requires_grad=True)
h = v.register_hook(f1)
v.backward(torch.tensor([0., 2., 3.]))
v1=v.grad
h.remove()

v2= torch.tensor([0., 0., 0.], requires_grad=True)
v2.backward(torch.tensor([0., 2., 3.]))
v3=v2.grad

v1,v3


# In[ ]:


# So here Gradient is always greater than zero gradients
def f1(grad):
    for i in range(grad.numel()):
        if grad[i]<0:
            grad[i]*=-1
        elif grad[i]==0:
            grad[i]+=1

v = torch.tensor([0., 1., 0.], requires_grad=True)
h = v.register_hook(f1)
v.backward(torch.tensor([-5., 0., 3.]))
v1=v.grad
h.remove()

v2= torch.tensor([0., 1., 0.], requires_grad=True)
v2.backward(torch.tensor([-5., 0., 3.]))
v3=v2.grad

v1,v3


# In[ ]:


v=torch.tensor([3.,3.,3.,3.], requires_grad=True)
learning_rate=1
h=v.register_hook(lambda grad:grad*learning_rate)
v.backward(torch.tensor([1.,1.5,3.,6.]))
v.grad


# In[ ]:


v=torch.tensor([3.,3.,3.,3.], requires_grad=True)
learning_rate=0.1
h=v.register_hook(lambda grad:grad*learning_rate)
v.backward(torch.tensor([1.,1.5,3.,6.]))
v.grad


# In[ ]:


# Breaking The Function
v=torch.tensor([3.,3.,3.,3.], requires_grad=True)
learning_rate=0.1
h=v.register_hook(lambda x:x[0]*learning_rate)
v.backward(torch.tensor([1.,1.5,3.,6.]))
v.grad


# In the above example we see that the function/Lambda function must not return only one value from the "grad" attribute of the 
# tensor. Rather return the whole "grad" attribute or return nothing as shown in the above examples. 

# Using register_hook() while doing Gradient Descent we can include the calculation of learning rate in the calcualtion of Gradient 
# for Backpropagation

# # Function 2
# ## **torch.Tensor.to_sparse(sparseDims)**
# 
# Returns a sparse copy of the tensor. We can optionally specify number of sparse dimensions using the "sparseDims" keyword.

# In the following example we see that we have a sparse matrix let's understand the attributes inside it
# 
# Inside the outer most tensor we have two tensors, indicies and values, alongwith size of the input and the number of non-zero 
# elements in the input  tensor.
# 
# So Indicies Tensor has always 2 rows, unless mentioned otherwise in "sparseDims".
# 
# The first row has the row indicies of non-zero elements.
# 
# THe second row has the column indicies of non-zero elements

# In[ ]:


s=torch.tensor(
    [
        [0,0,0],
        [4,0,0],
        [0,1,0]
    ]
)
s.to_sparse()


# So in the following example, in the same tensor we have restricted the indicies to 1 D therefore only the row indicies of the non-zero elements are present in indicies.
# 
# The values will have the original column order i.e. the whole row is present in the values tensor where there is a non-zero element present. 

# In[ ]:


s.to_sparse(sparse_dim=1)


# So sparse_dim cannot be greater than the dimension of the input tensor 

# In[ ]:


s.to_sparse(3)


# In the following example we can give maximum value of sparse_dim=3 i.e. the maximum dimension of the input

# In[ ]:


s1=torch.tensor(
    [
        [
            [0,2,0],
            [4,0,0],
            [0,0,0]
        ],
        [
            [0,0,12],
            [0,0,0],
            [16,0,0]
        ],
        [
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]
    ]
)

s1.to_sparse(3)


# We deal with sparse matrix in many areas of Machine Learning and Deep Learning a few are listed below:
#     
#     1) In OneHotEncoding
#     2) In data for BagOfWord model of a input Document.
#     3) Tf-Idf vectorizer    
#     4) some datasets while dealing with Recommender System. 

# # Function 3
# ## **torch.Tensor.dot(input, tensor)**
# 
# Computes the dot product of two 1D tensors only.

# In[ ]:


t=torch.tensor([1,2,3])
t1=torch.tensor([1,0,0])
t2=torch.tensor([0,1,0])
t3=torch.tensor([0,0,1])


# In[ ]:


(torch.Tensor.dot(t,t), # 1*1 + 2*2 + 3*3
 torch.Tensor.dot(t,t1), # 1*1 + 0*2 + 0*3
 torch.Tensor.dot(t,t2), # 0*1 + 1*2 + 0*3
 torch.Tensor.dot(t,t3)) # 0*1 + 0*2 + 1*3


# If we provide any tensors with dimension higher than 1 then the function will not accept.

# In[ ]:


ten=torch.tensor([
    [0,1],
    [2,3]
])
torch.Tensor.dot(ten,ten)


# Dot product is useful in many places but usually input would have dimension greater than 1. So this function can be of use but rarely. Although the numpy's dot product function can be used with any two matrix till they are compactible for the dot product to occur.

# # Function 4
# ## **torch.Tensor.flatten(input, start_dim=0, end_dim=-1)**
# 
# This function will flatten(reduce the dimensions) a continous range of dimensions in a tensor.
# It will return the modified tensor.

# In[ ]:


ten_sor=torch.tensor([
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,14,15],
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
])
ten_sor.shape


# The above tensor is a 3D tensor which has 3 dimensions 0,1,2 .
# 
# Also can be represented as -3 for 0, -2 for 1, -1 for 2
# 
# start_dim is the first dimension to flatten
# 
# end_dim is the last dimension to flatten
# 
# Both these indicate the range of dimensions which will be left flattened by the function any other dimension will be remain untouched.
# 
# Only consecutive dimensions can be flattened.

# In[ ]:


(
    torch.Tensor.flatten(ten_sor,start_dim=1,end_dim=2).shape,# 0th dimension will be untouched
    torch.Tensor.flatten(ten_sor,start_dim=0,end_dim=1).shape # 2nd dimesion will be untouched
)


# In[ ]:


torch.Tensor.flatten(ten_sor,start_dim=1,end_dim=2) 

#same as torch.flatten(ten_sor,start_dim=-2,end_dim=-1) 


# In[ ]:


torch.Tensor.flatten(ten_sor,start_dim=0,end_dim=1)

#same as torch.flatten(ten_sor,start_dim=-3,end_dim=-2)


# In[ ]:


# default #same as torch.flatten(ten_sor,start_dim=-3,end_dim=-1) or 
# same as torch.flatten(ten_sor,start_dim=0 ,end_dim=2)
torch.Tensor.flatten(ten_sor)


# We can have Problems with flatten function when start_dim comes after end_dim
# 
# As follows:

# In[ ]:


torch.Tensor.flatten(ten_sor,start_dim=2,end_dim=0)


# Function will break if the arguements have out of range values

# In[ ]:


torch.Tensor.flatten(ten_sor,start_dim=3)


# Flatten function is useful during processing of images with the help of Neural Networks (No CNN as CNN can take input as the whole image 2D/3D tensor but Neural Network cannot take a 2D/3D tensor but only 1D tensor)

# # Function 5
# ## **torch.Tensor.numel(input)**
# 
# Returns the total number of elements in the input tensor.

# In[ ]:


te=torch.tensor([
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,14,15],
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
])


# In[ ]:


te.numel()


# In[ ]:


torch.arange(10).numel()


# In[ ]:


torch.randn((10,101,8)).numel()


# numel() is a basic function which will be used in various places especially in for in loops.

# ## Conclusion
# 
# So we started from creating basic tensors using the PyTorch library and saw 5 functions which are important and are useful in many situations. These are the ones that caught my eye while reading the documentation and so I have tried to find out how exactly they work and in which situations we can use them.

# ## Reference Links
# * Official documentation for `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.commit()


# In[ ]:




