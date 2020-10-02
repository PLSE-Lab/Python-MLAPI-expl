#!/usr/bin/env python
# coding: utf-8

# # Exploring the world's coolest tensor using PyTorch functions.
# 
# For this assignment, My choice of 5 functions are 
# - torch.tensor.item()
# - torch.new_tensor(data, dtype=None, device=None, requires_grad=False)    
# - torch.exp(input, out=None)         
# - torch.sigmoid(input, out=None)     
# - torch.std_mean(input, dim, unbiased=True, keepdim=False)    

# In[ ]:


# Import torch and other required modules
import torch
import numpy as np


# ## Function 1 - torch.tensor.item() 
# This function is used to retrieve a scalar quantity from a single valued tensor, in other                         words, it is used to convert a single valued tensor to python number.

# In[ ]:


# Example 1 
x = torch.tensor([[3.14]])
y = x.item()

print(x.dtype)
print(type(y))


# As we can see above a tensor with single value is converted into a float data-type in python

# In[ ]:


# Example 2
x = torch.tensor([[1, 2], [3, 4]])
y = x.item()


# This above example shows that only a single valued tensor can be converted to a python number.

# ## Function 2 - torch.new_tensor(data, dtype=None, device=None, requires_grad=False)    
# 
# This function is used to intialize a pytorch tensor from a pre-existing python list.

# In[ ]:


# Example 1 - 
tensor = torch.zeros((2,), dtype=torch.float64)
x = [[2, 6], [26, 1728]]
y = tensor.new_tensor(x)
print(tensor)
print(y)
print(tensor.shape)
print(y.shape)


# As we can see in above example, whatever be dimension the initial tensor, we can change it according to the python list, also we can see that the data-type of element when copied to the tensor, gets changed according to the data-type tensor is previously created with.

# In[ ]:


tensor = torch.zeros((2,), dtype=torch.float64)
x = [[2, 6], [26, 1729]]
y = tensor.new_tensor(x)
print("Before changing value at x[1][1] : ")
print(y)
x[1][1] = 1.618
print("After changing value at x[1][1] : ")
print("x : ", x)
print("y : ", y)


# The above example we can see that new_tensor() copies the value from a python list to self. If you want to avoid this use torch.from_numpy()

# In[ ]:


tensor = torch.zeros((2,), dtype=torch.float64)
x = np.array([[2., 6.], [26., 1729.]])
y = torch.from_numpy(x)
print("Before changing value at x[1][1] : ")
print(y)
x[1][1] = 1.618
print("After changing value at x[1][1] : ")
print("x : ", x)
print("y : ", y)


# Here we see that we changed the value of x[1][1] from 1729 to 1.618 it is reflected in both, that is, the tensor y and the numpy array x. 

# ## Function 3 - torch.exp(input, out=None)         
#        
# This function is used to return the exponential value of elements in a tensor. Highly useful in calculating softmax function

# In[ ]:


x = torch.tensor([[11.2, 6.022], [1.380, 6.674]])
y = torch.exp(x)
print(y)


# In the above example we can see this fucntion calculates the exponential value of the given tensor. this function along with torch.sum() can be used to create one-liner softmax function used in multi-class classification (easy-peesy). To read more about it check out the reference below!

# ## Function 4 - torch.sigmoid(input, out=None)     
# 
# This function is used to calculate elemet-wise sigmoid value. Used in logistic regression.   

# In[ ]:


x = torch.tensor([[11.2, 6.022], [1.380, 6.674]])
y = torch.sigmoid(x)
print(y)


# This function is used to calculate sigmod value of the elements in a tensor. Sigmoid function in simple terms is used to place a number's value between 0 to 1. It is useful in logistic regression. To read more about it follow the reference below!

# ## Function 5 - torch.std_mean(input, dim, unbiased=True, keepdim=False)    
# 
# This function calculates and returns standard deviation and mean of all the values in a tensor

# In[ ]:


x = torch.tensor([1.380, 6.022, 1.602, 6.674])
torch.std_mean(x)


# As we can see in above example, to quickly calculate standard devation and mean of a tensor. we can use this function.

# In[ ]:


x = torch.tensor([[2,3,6,26], [1.380, 6.022, 1.602, 6.674]])
torch.std_mean(x,1)


# for a tensor whose dimension is greater than (x,1) or in this case a 2 dimensional tensor (matrix), if we pass dim = 1 it calculates the value value of standard deviation and mean by each column.

# In[ ]:


x = torch.tensor([2,3,6,26])
torch.std_mean(x)


# As we can see in above example, this function only accepts floating point (decimal) numbers.

# ## Conclusion
# 
# The function that we saw above are one of the most commonly used function and very powerful for performing complex things with fewer and most of the time single line of code.

# ## Reference Links
# 
# * Official documentation for `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html
# * Softmax function : https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
# * Multi-class classification : https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b
# * Logistic regression : https://medium.com/greyatom/logistic-regression-89e496433063

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.commit()


# In[ ]:




