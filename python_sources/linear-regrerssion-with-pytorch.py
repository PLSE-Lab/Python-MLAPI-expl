#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


#importing Numpy and PyTorch
import numpy as np
import torch


# ### Linear Regression with Gradiant descent method.

# Training Data

# In[ ]:


# Input(temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Output(apples, orange)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')


# In[ ]:


# inputs and targets into tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs , targets)


# ### Linear Regression
# Weights and biases initialized as random values.
# First row of weigths and first element of bias are used to predict first target variable.

# In[ ]:


# weights and biases
w = torch.randn(2,3, requires_grad = True)
b = torch.randn(2, requires_grad = True)
print(w,b)


# In[ ]:


def model(x):
    return x @ w.t() +b  # @ represents matrix multiplication in Pytorch.

preds = model(inputs)
preds


# In[ ]:


#compare preds with the targets
print(targets)


# **Loss Function**

# In[ ]:


# Meas Squarred Error
def mse(t1,t2):
    diff = (t1-t2)
    return torch.sum(diff**2)/diff.numel()

#Calculate loss b/wprediction and targets
loss = mse(preds,targets)
loss


# **Compute gradients**

# In[ ]:


#Compute gradients
loss.backward()

# Gradiants for weoghts
print(w.grad)
print(b.grad)


# In[ ]:


# weights and biases
print(w)
print(b)


# In[ ]:


# adjust weights and reset gradiants
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()
    


# In[ ]:


# new weights and biases
print(w)
print(b)


# In[ ]:


# calculate loss
preds= model(inputs)
loss = mse(preds,targets)
loss


# **Training the model for more epochs**

# In[ ]:


# Training for number of epochs to reduce the loss.
for i in range(3000):
    preds = model(inputs)
    loss = mse(preds,targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()


# In[ ]:


#Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)


# In[ ]:


print(preds)
print(targets)


# ### Linear Regression With Pytorch Builtins

# In[ ]:


#importing the torch.nn package from PyTorch, which contains utility classes for building neural networks
import torch.nn as nn

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# ### Dataset and DataLoader
# We'll create a TensorDataset, which allows access to rows from inputs and targets as tuples, and provides standard APIs for working with many different types of datasets in PyTorch

# In[ ]:


from torch.utils.data import TensorDataset, DataLoader


# In[ ]:


# Dataset 
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]


# In[ ]:


#DataLoader
batchsize = 5
train_dl = DataLoader(train_ds, batchsize ,shuffle= True)


# In[ ]:


#one of the batch
for xb, yb in train_dl:
    print(xb)
    print(yb)
    break


# ### nn.Linear
# Instead of initializing the weights & biases manually, we can define the model using the `nn.Linear` class from PyTorch, which does it automatically.

# In[ ]:


# define Model
model = nn.Linear(3,2)   # 3 input variables amd 2 output variables.
print(model.weight)
print(model.bias)


# PyTorch models also have a helpful `.parameters` method, which returns a list containing all the weights and bias matrices present in the model. For our linear regression model, we have one weight matrix and one bias matrix

# In[ ]:


# Parameters
list(model.parameters())


# In[ ]:


preds = model(inputs)
preds


# Loss Function

# In[ ]:


import torch.nn.functional  as F

# loss function
loss_fn= F.mse_loss

#compute loss
loss = loss_fn(preds,targets)
loss


# ### Optimizer
# Instead of manually manipulating the model's weights & biases using gradients, we can use the optimizer `optim.SGD`. SGD stands for `stochastic gradient descent`. It is called stochastic because samples are selected in batches (often with random shuffling) instead of as a single group.

# In[ ]:


#define optimizer
opt= torch.optim.SGD(model.parameters(), lr = 1e-5)


# ### Train the model
# We are now ready to train the model. We'll follow the exact same process to implement gradient descent:
# 
# - Generate predictions
# - Calculate the loss
# - Compute gradients w.r.t the weights and biases
# - Adjust the weights by subtracting a small quantity proportional to the gradient
# - Reset the gradients to zero

# In[ ]:


def fit(epoches, model, loss_fn, opt, train_dl):
    for epoch in range(epoches):
        for xb,yb in train_dl:
            pred=model(xb)
            
            # compute loss
            loss=loss_fn(pred,yb)
            
            # compute gradients
            loss.backward()
            
            # update parameter using gradients
            opt.step()
            
            #reset gradient to zero
            opt.zero_grad()
            
        if (epoch+1) %10 ==0:
            print(f"Epoch[{epoch+1}/{epoches}], Loss: {loss.item()} ")


# In[ ]:


fit(500, model, loss_fn, opt,train_dl)


# In[ ]:


preds = model(inputs)
preds


# In[ ]:


targets


# **Reference** :- I am practicing this notebook from Lecture 1 [Deep Laering with Pytorch Zero to GANS] on jovian.ml  https://jovian.ml/forum/c/pytorch-zero-to-gans/18

# In[ ]:




