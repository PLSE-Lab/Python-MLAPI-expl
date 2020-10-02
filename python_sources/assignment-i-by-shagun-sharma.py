#!/usr/bin/env python
# coding: utf-8

# # Assignment-I Five Interesting Pytorch Functions
# 
# ### By Shagun Sharma
# 
# An short introduction about PyTorch and about the chosen functions. 
# - function 1: torch.tensor()
# - function 2: torch.Tensor.size()
# - function 3: torch.Tensor.fill_diagonal_()
# - function 4: torch.Tensor.item()
# - function 5: torch.chunk()

# In[ ]:


# Import torch and other required modules
import torch


# In[ ]:


project_name='01-pytorch basics-5 important functions'


# ## Function 1 - torch.tensor ()
# 
# Pytorch uses Tensors, which are Multi-dimensional matrices of a particular datatype. This function creates a pytorch tensor.

# In[ ]:


# Example 1 - working (change this)
tens1=torch.tensor([[1, 2, 3], [4, 5, 6],[7, 8, 9]])


# In[ ]:


print(tens1)


# Function1 creates a 3X3 pytorch tensor stored in variable 'tens1' whose values are diaplayed in the above cell.

# In[ ]:


# Example 2 - working
tens2=torch.tensor([[1, 0], [0, 1]])
print(tens2)


# Function 1 creates a 2X2 identity pytorch tensor stored in variable tens2

# In[ ]:


# Example 3 - breaking (to illustrate when it breaks)
tens3=torch.tensor([[1, 2], [3, 4, 5]])
print(tens3)


# The function breaks when an element is added to the tensor that does not satisfy the dimensions of the tensor/matrix.

# Closing comments about when to use this function

# ## Function 2 - torch.Tensor.size()
# 
# This function displays the size(dimensions) of the pytorch tensor.

# In[ ]:


# Example 1 - working
tens1=torch.tensor([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
tens1.size()


# Function 2 displays the size of the pytorch tensor stored in the variable 'tens1'.

# In[ ]:


# Example 2 - working
torch.tensor([[1, 0, 0], [0, 1, 0]]).size()


# Function 2 displays the dimensions of the tensor.

# In[ ]:


# Example 3 - breaking (to illustrate when it breaks)
tens3=torch.tensor([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
tens3.size()


# In[ ]:


tens3.size(2)


# The function breaks when the argument of the torch.Tensor.size() is not in accordance with the domain.

# This function can be used where we need to know the dimensions of the tensors in use for example: Multiplication of two Tensors.

# ## Function 3 - torch.Tensor.fill_diagonal_()
# 
# This function is used to fill the diagonals of the the tensor with a particular value.

# In[ ]:


# Example 1 - working
tens1=torch.zeros(3, 3)
print(tens1)


# In[ ]:


tens1.fill_diagonal_(1)


# This function fills in the value specified as its argument(i.e. 1) as the diagonal value of the tensor(tens1).

# In[ ]:


# Example 2 - working
torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]])


# In[ ]:


torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]]).fill_diagonal_(0)


# Function can be directly applied on the tensor or by using a variable as in previous example.

# In[ ]:


# Example 3 - breaking (to illustrate when it breaks)
tens2=torch.tensor([[[1, 2, 3, 4]],[[5, 6, 7, 8]]])
print(tens2)
tens2.size()


# In[ ]:


tens2.fill_diagonal_(10)


# The matrix dimension does not satisfy the condition of the diagonal matrix to be imposed with a value.

# This function can be used at any relevant place where a particular diagonal value is to be imposed in the tensor. Such applications might use the concept of upper and lower triangle matrices where this function proves to be helpful in visually segregating the same.

# ## Function 4 - torch.Tensor.item()
# 
# We can use this function to get a number from a tensor containing a single value.

# In[ ]:


# Example 1 - working
tens1=torch.tensor([1.7])
print(tens1)
tens1.item()


# Function is called on a tensor containing single value and stored in the variable tens1.

# In[ ]:


# Example 2 - working
torch.tensor([2]).item()


# Function is directly called over a tensor containing a single value 2.

# In[ ]:


# Example 3 - breaking (to illustrate when it breaks)
tens2=torch.tensor([[1, 2, 3],[4, 5, 6]])
print(tens2)
tens2.item()


# Function fails when the tensor is not a single valued tensor.

# Function can be generally used to instantiate a single value wherever necessary.

# ## Function 5 - torch.chunk()
# 
# This function can be used to divide a tensor into specified equal parts.

# In[ ]:


# Example 1 - working
tens1=torch.tensor([[1.1, 1.2, 1.3],[1.4, 1.5, 1.6]])
print(tens1)
torch.chunk(tens1,2,1)


# The function divides the tensor into 2 chunks based on the dimension specified(i.e. 1 here).

# In[ ]:


# Example 2 - working
a,b,c=torch.chunk(torch.tensor([1, 2, 3, 4, 5, 6]),3,0)
a,b,c


# Function divides the tensor into specified chunks and also assigns the values into variables.

# In[ ]:


# Example 3 - breaking (to illustrate when it breaks)
tens2=torch.tensor([1, 2, 3, 4, 5, 6])
torch.chunk(tens2,3,1)


# Function breaks wherever the dimension of the tensor is specified incorrectly as the argument.

# This function can be used wherever a division of the tensor is to be made in the program.

# ## Conclusion
# 
# We discussed 5 different pytorch functions in this notebook which are seemingly relevant in many programming tasks that require some modification of the tensors since all the data in pytorch is computed using tensors. Hence these functions are few of the many functions that could be used to brush our hands up on pytorch

# ## Reference Links
# Provide links to your references and other interesting articles about tensors
# * Official documentation for `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html
# * ...

# In[ ]:


get_ipython().system(' pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.commit(project=project_name)


# In[ ]:




