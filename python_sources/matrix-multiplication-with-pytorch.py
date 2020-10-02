#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import torch


# # Matrix Multiplication 
# is a basic building block which is used almost all the time while doing deep learning. Here we will implement a basic version of this.

# MATMUL - following is implemented using purely python.

# In[ ]:


def matmul(a,b):
    arow, acol = a.shape
    brow, bcol = b.shape
    c = torch.zeros(arow, bcol) # creating the output array
    for i in range(arow):
            #print(i)
            for j in range(acol):
                    for k in range(bcol):
                        c[i,k] = a[i,j] * b[j,k]
    return(c)


# In[ ]:


x = torch.randn(2,1)
y = torch.randn(1,2)


# In[ ]:


z = matmul(x,y)


# In[ ]:


print(z)


# In[ ]:


def matmul(a,b):
    arow, acol = a.shape
    brow, bcol = b.shape
    c = torch.zeros(arow, bcol) # creating the output array
    for i in range(arow):
            #print(i)
            for j in range(acol):
                for k in range(bcol):
                    c[i,k] = (a[i,:] * b[:,k]).sum()
    return(c)


# Above can be made faster by getting rid of the third loop. In general, we should try to avoid loops wherever possible. Code above is the version 2 of matmul where we do entire row/column operation once.

# In[ ]:


z = matmul(x,y)
print(z)
# The output should confirm to earlier one.


# We will now use broadcasting which is even faster. Here we are taking one more loop off with help of boradcasting.

# In[ ]:


def matmul(a,b):
    arow, acol = a.shape
    brow, bcol = b.shape
    c = torch.zeros(arow, bcol) # creating the output array
    for i in range(arow):
        c[i,:] = (a[i,None] * b).sum(dim = 0)
    return(c)


# In[ ]:


print(matmul(x,y))


# So, we are down from three loops to one loop using row/column operations in one go instead element wise and using boradcasting. This is using Einstein Summation or **EinSum**. Two same letters on different inputs means it does a dot product for us. 

# In[ ]:


def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)


# In[ ]:


matmul(x,y)


# That's it for now. Thanks for reading and please do send your feedback through the comments section.
# 
# Note: This Kernel is produced by following Jeremy Howard's FastAI course. You can check this out at fast.ai
# 
