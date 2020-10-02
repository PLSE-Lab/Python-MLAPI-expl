#!/usr/bin/env python
# coding: utf-8

# ## Contents
# - Getting Started with PyTorch
# - Loading and processing data
# - Building Models
# - Loss function and optimizer
# - Training and evaluation

# ## Overview
# Deep learning is a field of machine learning utilizing massive neural networks, massive datasets, and accelerated computing on GPUs. Many of the advancements we've seen in AI recently are due to the power of deep learning. This revolution is impacting a wide range of industries already with applications such as personal voice assistants, medical imaging, automated vehicles, video game AI, and more.
# 
# In this notebook, I'll be covering the concepts behind deep learning and how to build deep learning models using PyTorch. By the end of the notebook, you'll be defining and training your own state-of-the-art deep learning models.

# ## What is Machine Learning?
# ML is a subfield of AI that uses algorithms and statistical techniques to perform a task without the use of any explicit instructions. Instead, it relies on underlying statistical patterns in the data.

# 
# ## What is CUDA?
# 
# CUDA is a framework developed by NVIDIA that allows us to use General Purpose Computing on Graphics Processing Units (GPGPU). It is a widely used framework written in C++ that allows us to write general-purpose programs that run on GPUs. Almost all deep learning frameworks leverage CUDA to execute instructions on GPUs

# ## PyTorch
# [PyTorch](https://pytorch.org) is an open-source Python framework from the [Facebook AI Research team](https://ai.facebook.com/) used for developing deep neural networks. PyTorch as an extension of Numpy that has some convenience classes for defining neural networks and accelerated computations using GPUs. PyTorch is designed with a Python-first philosophy, it follows Python conventions and idioms, and works perfectly alongside popular Python packages.

# ## What is a neural network?
# 
# In short, a neural network is an algorithm that learns about the relationship between the input variables and their associated target variables.

# ## Getting Started with PyTorch

# ### Verifying the installation
# 
# Let's make sure that the installation is correct by importing PyTorch into Python:

# In[ ]:


# import PyTorch
import torch
# import torchvision
import torchvision

# get PyTorch version
print(torch.__version__)
# get torchvision version
print(torchvision.__version__)


# In[ ]:


# checking if cuda is available
torch.cuda.is_available()


# In[ ]:


# get number of cuda/gpu devices
torch.cuda.device_count()


# In[ ]:


# get cuda/gpu device id
torch.cuda.current_device()


# In[ ]:


# get cuda/gpu device name
torch.cuda.get_device_name(0)


# Tensors are data containers and are a generalized representation of vectors and matrices. A vector is a first-order tensor since it only has one axis and would look like [x1, x2, x3,..]. A matrix is a second-order tensor that has two axes and looks like [ [x11, x12, x13..] , [x21, x22, x23..] ]. On the other hand, a scalar is a zero-order tensor that only contains a single element, such as x1. 

# In[ ]:


x = torch.rand(2,3)
print(x)


# ### Working with PyTorch tensors
# PyTorch is built on tensors. A PyTorch tensor is an n-dimensional array, similar to NumPy arrays.
# 
# If you are familiar with NumPy, you will see a similarity in the syntax when working with tensors, as shown in the following table:
# 
# |NumPy Arrays |	PyTorch tensors |	Description|
# |-----------------|-----------------|-------------------|
# |numpy.ones(.) |	torch.ones(.) |	Create an array of ones|
# |numpy.zeros(.) |	torch.zeros(.) |	Create an array of zeros|
# |numpy.random.rand(.) |	torch.rand(.) |	Create a random array|
# |numpy.array(.) |	torch.tensor(.) |	Create an array from given values|
# |x.shape |	x.shape or x.size() |	Get an array shape|

# ### Defining the tensor data type
# The default tensor data type is `torch.float32`. This is the most used data type for tensor operations.

# In[ ]:


# Define a tensor with a default data type:
x = torch.ones(2,3)
print(x)
print(x.dtype)


# In[ ]:


# define a tensor with specific data type
x = torch.ones(2,3,dtype=torch.int16)
print(x)
print(x.dtype)


# ### Changing the tensor's data type
# We can change a tensor's data type using the `.type` method:

# In[ ]:


x = torch.ones(3,3,dtype=torch.int8)
print(x.dtype)
# Change the tensor datatype
x = x.type(torch.float32)
print(x.dtype)


# ### Create a tensor filled with a specific value

# In[ ]:


s_val = torch.full((3,4),3.1416)
print(s_val)


# ### Create an empty tensor

# In[ ]:


e_val = torch.empty((3,4))
print(e_val)


# ### Create a tensor with mean 0 and variance 1

# In[ ]:


r_val = torch.randn((4,5))
print(r_val)


# ### Create a tensor from given range of values

# In[ ]:


rng_val = torch.randint(10,20,(3,4))
print(rng_val)


# ### Converting tensors into NumPy arrays
# We can easily convert PyTorch tensors into NumPy arrays using `.numpy` method.

# In[ ]:


# Define a tensor
x = torch.rand(2,3)
print(x)
print(x.dtype)
# convert tensor into numpy array
y = x.numpy()
print(y)
print(y.dtype)


# ### Converting NumPy arrays into tensors
# We can also convert NumPy arrays into PyTorch tensors using `.from_numpy` method.

# In[ ]:


import numpy as np
# define a numpy array
x = np.ones((2,3),dtype=np.float32)
print(x)
print(x.dtype)
# convert to pytorch tensor
y = torch.from_numpy(x)
print(y)
print(y.dtype)


# ### Moving tensors between devices
# 
# By default, PyTorch tensors are stored on the CPU. PyTorch tensors can be utilized on a GPU to speed up computing. This is the main advantage of tensors compared to NumPy arrays. To get this advantage, we need to move the tensors to the CUDA device. We can move tensors onto any device using the `.to` method:

# In[ ]:


# Define a tensor in cpu
x = torch.tensor([2.3,5.8])
print(x)
print(x.device)

# Define a CUDA device
if torch.cuda.is_available():
    device= torch.device("cuda:0")
    
# Move the tensor onto CUDA device
x = x.to(device)
print(x)
print(x.device)


# In[ ]:


# Similarly, we can move tensors to CPU:
# define a cpu device
device = torch.device("cpu")
x = x.to(device) 
print(x)
print(x.device)


# In[ ]:


# We can also directly create a tensor on any device:
# define a tensor on device
device = torch.device("cuda:0")
x = torch.ones(2,2, device=device) 
print(x)


# ## Loading and processing data
# In most cases, we receive data in three groups: training, validation, and test. We use the training dataset to train the model. The validation dataset is used to track the model's performance during training and test dataset used for the final evaluation of the model. The target values of the test dataset are usually hidden from us. We need at least one training dataset and one validation dataset to be able to develop and train a model.

# ### Loading a dataset
# The PyTorch `torchvision` package provides multiple popular datasets. Let's load the `MNIST` dataset from `torchvision`:

# In[ ]:


from torchvision import datasets
# path to store data and/or load from
data_path="./data"
# loading training data
train_data=datasets.MNIST(data_path, train=True, download=True)


# In[ ]:


# extract data and targets
x_train, y_train=train_data.data,train_data.targets
print(x_train.shape)
print(y_train.shape)


# In[ ]:


# loading validation data
val_data=datasets.MNIST(data_path, train=False, download=True)


# In[ ]:


# extract data and targets
x_val,y_val=val_data.data, val_data.targets
print(x_val.shape)
print(y_val.shape)


# In[ ]:


# add a dimension to tensor to become B*C*H*W
if len(x_train.shape)==3:
    x_train=x_train.unsqueeze(1)
print(x_train.shape)

if len(x_val.shape)==3:
    x_val=x_val.unsqueeze(1)
print(x_val.shape)


# Let's display a few sample images.

# In[ ]:


from torchvision import utils
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# define a helper function to display tensors as images
def show(img):
    # convert tensor to numpy array
    npimg = img.numpy()
    # Convert to H*W*C shape
    npimg_tr=np.transpose(npimg, (1,2,0))
    plt.imshow(npimg_tr,interpolation='nearest')

    
# make a grid of 40 images, 8 images per row
x_grid=utils.make_grid(x_train[:40], nrow=8, padding=2)
print(x_grid.shape)
# call helper function
show(x_grid)


# ### Data transformation
# Image transformation/augmentation is an effective technique that's used to improve a model's performance. The `torchvision` package provides common image transformations through the `transforms` class

# In[ ]:


from torchvision import transforms
# define transformations
data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
    ])


# In[ ]:


# get a sample image from training dataset
img = train_data[0][0]

# transform sample image
img_tr=data_transform(img)

# convert tensor to numpy array
img_tr_np=img_tr.numpy()

# show original and transformed images
plt.subplot(1,2,1)
plt.imshow(img,cmap="gray")
plt.title("original")
plt.subplot(1,2,2)
plt.imshow(img_tr_np[0],cmap="gray");
plt.title("transformed")


# We can also pass the transformer function to the dataset class:

# In[ ]:


# define transformations
# data_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(1),
#         transforms.RandomVerticalFlip(1),
#         transforms.ToTensor(),
#     ])

# Loading MNIST training data with on-the-fly transformations
# train_data=datasets.MNIST(path2data, train=True, download=True, transform=data_transform )


# **create a dataset from tensors.**

# In[ ]:


from torch.utils.data import TensorDataset

# wrap tensors into a dataset
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)

for x,y in train_ds:
    print(x.shape,y.item())
    break


# ### Creating data loaders
# 
# To easily iterate over the data during training, we can create a data loader using the `DataLoader` class

# In[ ]:


from torch.utils.data import DataLoader

# create a data loader from dataset
train_dl = DataLoader(train_ds, batch_size=8)
val_dl = DataLoader(val_ds, batch_size=8)

# iterate over batches
for xb,yb in train_dl:
    print(xb.shape)
    print(yb.shape)
    break


# ## Building Models
# A model is a collection of connected layers that process the inputs to generate the outputs. You can use the nn package to define models. The nn package is a collection of modules that provide common deep learning layers. A module or layer of nn receives input tensors, computes output tensors, and holds the weights, if any. There are two methods we can use to define models in PyTorch: nn.Sequential and nn.Module.

# Create a linear layer and print out its output size

# In[ ]:


from torch import nn

# input tensor dimension 64*1000
input_tensor = torch.randn(64, 1000) 
# linear layer with 1000 inputs and 10 outputs
linear_layer = nn.Linear(1000, 10) 
# output of the linear layer
output = linear_layer(input_tensor) 
print(output.size())


# In[ ]:


# implement and print the model using nn.Sequential
from torch import nn

# define a two-layer model
model = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(), 
    nn.Linear(5, 1),
)
print(model)


# ### Defining models using nn.Module
# 
#  Another way of defining models in PyTorch is by subclassing the nn.Module class. In this method, we specify the layers in the __init__ method of the class. Then, in the forward method, we apply the layers to inputs. This method provides better flexibility for building customized models.

# In[ ]:


import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
#     Net.__init__ = __init__
#     Net.forward = forward


# In[ ]:


model = Net()    
print(model)


# **Moving the model to a CUDA device**

# In[ ]:


print(next(model.parameters()).device)


# In[ ]:


device = torch.device("cuda:0")
model.to(device)
print(next(model.parameters()).device)


# In[ ]:


get_ipython().system('pip install torchsummary')


# In[ ]:


# model summary using torchsummary
from torchsummary import summary
summary(model, input_size=(1, 28, 28))


# ## Loss function and Optimizer

# **The negative log-likelihood loss:**

# In[ ]:


from torch import nn
loss_func = nn.NLLLoss(reduction="sum")


# In[ ]:


for xb, yb in train_dl:
    # move batch to cuda device
    xb=xb.type(torch.float).to(device)
    yb=yb.to(device)
    # get model output
    out=model(xb)
    # calculate loss value
    loss = loss_func(out, yb)
    print (loss.item())
    break


# In[ ]:


# define the Adam optimizer
from torch import optim
opt = optim.Adam(model.parameters(), lr=1e-4)


# In[ ]:


# update model parameters
opt.step()


# In[ ]:


# set gradients to zero
opt.zero_grad()


# 

# ## Training and evaluation

# In[ ]:


#  helper function to compute the loss value per mini-batch
def loss_batch(loss_func, xb, yb,yb_h, opt=None):
    # obtain loss
    loss = loss_func(yb_h, yb)
    
    # obtain performance metric
    metric_b = metrics_batch(yb,yb_h)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), metric_b


# In[ ]:


# helper function to compute the accuracy per mini-batch
def metrics_batch(target, output):
    # obtain output class
    pred = output.argmax(dim=1, keepdim=True)
    
    # compare output class with target class
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects


# In[ ]:


# helper function to compute the loss and metric values for a dataset
def loss_epoch(model,loss_func,dataset_dl,opt=None):
    loss=0.0
    metric=0.0
    len_data=len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb=xb.type(torch.float).to(device)
        yb=yb.to(device)
        
        # obtain model output
        yb_h=model(xb)

        loss_b,metric_b=loss_batch(loss_func, xb, yb,yb_h, opt)
        loss+=loss_b
        if metric_b is not None:
            metric+=metric_b
    loss/=len_data
    metric/=len_data
    return loss, metric


# In[ ]:


def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,opt)
  
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl)
        
        accuracy=100*val_metric

        print("epoch: %d, train loss: %.6f, val loss: %.6f, accuracy: %.2f" %(epoch, train_loss,val_loss,accuracy))


# In[ ]:


# call train_val function
num_epochs=5
train_val(num_epochs, model, loss_func, opt, train_dl, val_dl)


# ### Storing and loading models

# In[ ]:


# define path to weights
weights_path ="weights.pt"
 
# store state_dict to file
torch.save(model.state_dict(), weights_path)


# In[ ]:


# define model: weights are randomly initiated
_model = Net()
weights=torch.load(weights_path)
_model.load_state_dict(weights)
_model.to(device)


# ### Deploying the model

# In[ ]:


n=100
x= x_val[n]
y=y_val[n]
print(x.shape)
plt.imshow(x.numpy()[0],cmap="gray")


# In[ ]:


# we use unsqueeze to expand dimensions to 1*C*H*W
x= x.unsqueeze(0)

# convert to torch.float32
x=x.type(torch.float)

# move to cuda device
x=x.to(device)


# In[ ]:


# get model output
output=_model(x)

# get predicted class
pred = output.argmax(dim=1, keepdim=True)
print (pred.item(),y.item())


# In[ ]:




