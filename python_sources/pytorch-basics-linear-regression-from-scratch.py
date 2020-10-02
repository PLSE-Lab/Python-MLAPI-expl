#!/usr/bin/env python
# coding: utf-8

# # PyTorch basics - Linear Regression from scratch
# 
# <!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/ECHX1s0Kk-o?controls=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->
# 
# Tutorial inspired from [FastAI development notebooks](https://github.com/fastai/fastai_v1/tree/master/dev_nb)
# 
# ## Machine Learning
# 
# <img src="https://i.imgur.com/oJEQe7k.png" width="500">
# 
# 
# ## Tensors & Gradients

# In[ ]:


# Import Numpy & PyTorch
import numpy as np
import torch


# A tensor is a number, vector, matrix or any n-dimensional array.

# In[ ]:


# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)


# In[ ]:


# Print tensors
print(x)
print(w)
print(b)


# We can combine tensors with the usual arithmetic operations.

# In[ ]:


# Arithmetic operations
y = w * x + b
print(y)


# What makes PyTorch special, is that we can automatically compute the derivative of `y` w.r.t. the tensors that have `requires_grad` set to `True` i.e. `w` and `b`.

# In[ ]:


# Compute gradients
y.backward()


# In[ ]:


# Display gradients
print('dy/dw:', w.grad)
print('dy/db:', b.grad)


# ## Problem Statement

# We'll create a model that predicts crop yeilds for apples and oranges (*target variables*) by looking at the average temperature, rainfall and humidity (*input variables or features*) in a region. Here's the training data:
# 
# <img src="https://i.imgur.com/lBguUV9.png" width="500" />
# 
# In a **linear regression** model, each target variable is estimated to be a weighted sum of the input variables, offset by some constant, known as a bias :
# 
# ```
# yeild_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
# yeild_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2
# ```
# 
# Visually, it means that the yield of apples is a linear or planar function of the temperature, rainfall & humidity.
# 
# <img src="https://i.imgur.com/mtkR2lB.png" width="540" >
# 
# 
# **Our objective**: Find a suitable set of *weights* and *biases* using the training data, to make accurate predictions.

# ## Training Data
# The training data can be represented using 2 matrices (inputs and targets), each with one row per observation and one column per variable.

# In[ ]:


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')


# In[ ]:


# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')


# Before we build a model, we need to convert inputs and targets to PyTorch tensors.

# In[ ]:


# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)


# ## Linear Regression Model (from scratch)
# 
# The *weights* and *biases* can also be represented as matrices, initialized with random values. The first row of `w` and the first element of `b` are use to predict the first target variable i.e. yield for apples, and similarly the second for oranges.

# In[ ]:


# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)


# The *model* is simply a function that performs a matrix multiplication of the input `x` and the weights `w` (transposed) and adds the bias `b` (replicated for each observation).
# 
# $$
# \hspace{2.5cm} X \hspace{1.1cm} \times \hspace{1.2cm} W^T \hspace{1.2cm}  + \hspace{1cm} b \hspace{2cm}
# $$
# 
# $$
# \left[ \begin{array}{cc}
# 73 & 67 & 43 \\
# 91 & 88 & 64 \\
# \vdots & \vdots & \vdots \\
# 69 & 96 & 70
# \end{array} \right]
# %
# \times
# %
# \left[ \begin{array}{cc}
# w_{11} & w_{21} \\
# w_{12} & w_{22} \\
# w_{13} & w_{23}
# \end{array} \right]
# %
# +
# %
# \left[ \begin{array}{cc}
# b_{1} & b_{2} \\
# b_{1} & b_{2} \\
# \vdots & \vdots \\
# b_{1} & b_{2} \\
# \end{array} \right]
# $$

# In[ ]:


# Define the model
def model(x):
    return x @ w.t() + b


# The matrix obtained by passing the input data to the model is a set of predictions for the target variables.

# In[ ]:


# Generate predictions
preds = model(inputs)
print(preds)


# In[ ]:


# Compare with targets
print(targets)


# Because we've started with random weights and biases, the model does not a very good job of predicting the target varaibles.

# ## Loss Function
# 
# We can compare the predictions with the actual targets, using the following method: 
# * Calculate the difference between the two matrices (`preds` and `targets`).
# * Square all elements of the difference matrix to remove negative values.
# * Calculate the average of the elements in the resulting matrix.
# 
# The result is a single number, known as the **mean squared error** (MSE).

# In[ ]:


# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# In[ ]:


# Compute loss
loss = mse(preds, targets)
print(loss)


# The resulting number is called the **loss**, because it indicates how bad the model is at predicting the target variables. Lower the loss, better the model. 

# ## Compute Gradients
# 
# With PyTorch, we can automatically compute the gradient or derivative of the `loss` w.r.t. to the weights and biases, because they have `requires_grad` set to `True`.

# In[ ]:


# Compute gradients
loss.backward()


# The gradients are stored in the `.grad` property of the respective tensors.

# In[ ]:


# Gradients for weights
print(w)
print(w.grad)


# In[ ]:


# Gradients for bias
print(b)
print(b.grad)


# A key insight from calculus is that the gradient indicates the rate of change of the loss, or the slope of the loss function w.r.t. the weights and biases. 
# 
# * If a gradient element is **postive**, 
#     * **increasing** the element's value slightly will **increase** the loss.
#     * **decreasing** the element's value slightly will **decrease** the loss.
# 
# <img src="https://i.imgur.com/2H4INoV.png" width="400" />
# 
# 
# 
# * If a gradient element is **negative**,
#     * **increasing** the element's value slightly will **decrease** the loss.
#     * **decreasing** the element's value slightly will **increase** the loss.
#     
# <img src="https://i.imgur.com/h7E2uAv.png" width="400" />    
# 
# The increase or decrease is proportional to the value of the gradient.

# Finally, we'll reset the gradients to zero before moving forward, because PyTorch accumulates gradients.

# In[ ]:


w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)


# ## Adjust weights and biases using gradient descent
# 
# We'll reduce the loss and improve our model using the gradient descent algorithm, which has the following steps:
# 
# 1. Generate predictions
# 2. Calculate the loss
# 3. Compute gradients w.r.t the weights and biases
# 4. Adjust the weights by subtracting a small quantity proportional to the gradient
# 5. Reset the gradients to zero

# In[ ]:


# Generate predictions
preds = model(inputs)
print(preds)


# In[ ]:


# Calculate the loss
loss = mse(preds, targets)
print(loss)


# In[ ]:


# Compute gradients
loss.backward()


# In[ ]:


# Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()


# In[ ]:


print(w)


# With the new weights and biases, the model should have a lower loss.

# In[ ]:


# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)


# ## Train for multiple epochs
# 
# To reduce the loss further, we repeat the process of adjusting the weights and biases using the gradients multiple times. Each iteration is called an epoch.

# In[ ]:


# Train for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()


# In[ ]:


# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)


# In[ ]:


# Print predictions
preds


# In[ ]:


# Print targets
targets


# ## Linear Regression Model using PyTorch built-ins
# 
# Let's re-implement the same model using some built-in functions and classes from PyTorch.

# In[ ]:


# Imports
import torch.nn as nn


# In[ ]:


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')


# In[ ]:


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# ### Dataset and DataLoader
# 
# We'll create a `TensorDataset`, which allows access to rows from `inputs` and `targets` as tuples. We'll also create a DataLoader, to split the data into batches while training. It also provides other utilities like shuffling and sampling.

# In[ ]:


# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader


# In[ ]:


# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]


# In[ ]:


# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
next(iter(train_dl))


# ### nn.Linear
# Instead of initializing the weights & biases manually, we can define the model using `nn.Linear`.

# In[ ]:


# Define model
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)


# ### Optimizer
# Instead of manually manipulating the weights & biases using gradients, we can use the optimizer `optim.SGD`.

# In[ ]:


# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# ### Loss Function
# Instead of defining a loss function manually, we can use the built-in loss function `mse_loss`.

# In[ ]:


# Import nn.functional
import torch.nn.functional as F


# In[ ]:


# Define loss function
loss_fn = F.mse_loss


# In[ ]:


loss = loss_fn(model(inputs), targets)
print(loss)


# ### Train the model
# 
# We are ready to train the model now. We can define a utility function `fit` which trains the model for a given number of epochs.

# In[ ]:


# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))


# In[ ]:


# Train the model for 100 epochs
fit(100, model, loss_fn, opt)


# In[ ]:


# Generate predictions
preds = model(inputs)
preds


# In[ ]:


# Compare with targets
targets


# # Bonus: Feedfoward Neural Network
# 
# ![ffnn](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Multi-Layer_Neural_Network-Vector-Blank.svg/400px-Multi-Layer_Neural_Network-Vector-Blank.svg.png)
# 
# Conceptually, you think of feedforward neural networks as two or more linear regression models stacked on top of one another with a non-linear activation function applied between them.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*XxxiA0jJvPrHEJHD4z893g.png" width="640">
# 
# To use a feedforward neural network instead of linear regression, we can extend the `nn.Module` class from PyTorch.

# In[ ]:


class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(3, 2)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


# Now we can define the model, optimizer and loss function exactly as before.

# In[ ]:


model = SimpleNet()
opt = torch.optim.SGD(model.parameters(), 1e-5)
loss_fn = F.mse_loss


# Finally, we can apply gradient descent to train the model using the same `fit` function defined earlier for linear regression.
# 
# <img src="https://i.imgur.com/g7Rl0r8.png" width="500">

# In[ ]:


fit(100, model, loss_fn, opt)


# In[ ]:




