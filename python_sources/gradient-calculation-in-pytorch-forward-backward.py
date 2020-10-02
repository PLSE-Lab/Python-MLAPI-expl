#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch


# ## Calculate Gradient

# In[ ]:


x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
print(z)

z = z.mean()
print(z)

# Calculate the gradient
z.backward() # dz/dx
print(x.grad)


# ## If the input are not scalar

# In[ ]:


x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
print(z)

# Because backward is Vector Jacobian Product, you need a vector ! 
v = torch.tensor([0.1,1.0,0.001], dtype = torch.float32)

# Calculate the gradient
z.backward(v) # dz/dx
print(x.grad)


# ## 3 Ways to prevent for tracking the gradients
# 
# ```python
# 1. x.requires_grad_(False) # Inplacing the gradient to false
# 2. x.detach() # Create new tensor that doesn't required the gradient
# 3. with torch.no_grad(): 
#     pass
# ```

# In[ ]:


print(x)

# Prevent the gradient to calculate
v = x.detach()
b = x.requires_grad_(False)
# x.requires_grad_(False)

print(b)
print(v)

# Prevent the gradient to calculate
with torch.no_grad():
    g = y+2
    print(g)

# Still calculate the gradient
g = y+2 
print(g)


# ## grad in looping

# In[ ]:


weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    
    model_output.backward()
    print(weights.grad)
    
    #weights.grad.zero_() # You need to re-assgin to zero
    #print(weights.grad)


# ## Backpropagation

# In[ ]:


x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat-y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)

## update weights
## next forwards and backwards


# In[ ]:




