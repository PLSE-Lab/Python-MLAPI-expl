#!/usr/bin/env python
# coding: utf-8

# In the previous tutorial, we trained a logistic regression model to identify handwritten digits from the MNIST dataset with an accuracy of around 86%.
# 
# ![1_q2nyeRU8uvjPeKpo_lMXgQ.jpeg](attachment:1_q2nyeRU8uvjPeKpo_lMXgQ.jpeg)
# 
# However, we also noticed that it's quite difficult to improve the accuracy beyond 87%, due to the limited power of the model. In this post, we'll try to improve upon it using a feedforward neural network.

# # Preparing the Data
# 
# The data preparation is identical to the previous tutorial. We begin by importing the required modules & classes.

# In[ ]:


# Uncomment and run the commands below if imports fail
# !conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y
# !pip install matplotlib --upgrade --quiet


# In[ ]:


import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
get_ipython().run_line_magic('matplotlib', 'inline')


# We download the data and create a PyTorch dataset using the **MNIST** class from **torchvision.datasets**.

# In[ ]:


dataset = MNIST(root='data/', download=True, transform=ToTensor())


# Next, let's use the random_split helper function to set aside 10000 images for our validation set.

# In[ ]:


val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# We can now create PyTorch data loaders for training and validation.

# In[ ]:


batch_size=128


# In[ ]:


train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)


# Let's visualize a batch of data in a grid using the **make_grid** function from torchvision. We'll also use the **.permute** method on the tensor to move the channels to the last dimension, as expected by matplotlib.

# In[ ]:


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


# # Model
# 
# To improve upon logistic regression, we'll create a neural network with one hidden layer. Here's what this means:
# 
# * Instead of using a single nn.Linear object to transform a batch of inputs (pixel intensities) into a batch of outputs (class probabilities), we'll use two nn.Linear objects. Each of these is called a layer in the network.
# * The first layer (also known as the hidden layer) will transform the input matrix of shape batch_size x 784 into an intermediate output matrix of shape batch_size x hidden_size, where hidden_size is a preconfigured parameter (e.g. 32 or 64).
# * The intermediate outputs are then passed into a non-linear activation function, which operates on individual elements of the output matrix.
# * The result of the activation function, which is also of size batch_size x hidden_size, is passed into the second layer (also knowns as the output layer), which transforms it into a matrix of size batch_size x 10, identical to the output of the logistic regression model.
# 
# Introducing a hidden layer and an activation function allows the model to learn more complex, multi-layered and non-linear relationships between the inputs and the targets. Here's what it looks like visually:
# 
# ![vDOGEkG.png](attachment:vDOGEkG.png)
# 
# The activation function we'll use here is called a **Rectified Linear Unit** or **ReLU**, and it has a really simple formula: relu(x) = max(0,x) i.e. if an element is negative, we replace it by 0, otherwise we leave it unchanged.
# 
# To define the model, we extend the nn.Module class, just as we did with logistic regression.

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:


class MnistModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


# We'll create a model that contains a hidden layer with 32 activations.

# In[ ]:


input_size = 784
hidden_size = 32 # you can change this
num_classes = 10


# In[ ]:


model = MnistModel(input_size, hidden_size=32, out_size=num_classes)


# Let's take a look at the model's parameters. We expect to see one weight and bias matrix for each of the layers.

# In[ ]:


for t in model.parameters():
    print(t.shape)


# Let's try and generate some outputs using our model. We'll take the first batch of 128 images from our dataset, and pass them into our model.

# In[ ]:


for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)


# # Using a GPU
# 
# As the sizes of our models and datasets increase, we need to use GPUs to train our models within a reasonable amount of time. GPUs contain hundreds of cores that are optimized for performing expensive matrix operations on floating point numbers in a short time, which makes them ideal for training deep neural networks with many layers. You can use GPUs for free on **Kaggle** kernels or **Google Colab**, or rent GPU-powered machines on services like **Google Cloud Platform**, **Amazon Web Services** or **Paperspace**.
# 
# We can check if a GPU is available and the required NVIDIA CUDA drivers are installed using torch.cuda.is_available.

# In[ ]:


torch.cuda.is_available()


# Let's define a helper function to ensure that our code uses the GPU if available, and defaults to using the CPU if it isn't.

# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# In[ ]:


device = get_default_device()
device


# Next, let's define a function that can move data and model to a chosen device.

# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[ ]:


for images, labels in train_loader:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break


# Finally, we define a DeviceDataLoader class to wrap our existing data loaders and move data to the selected device, as a batches are accessed. Interestingly, we don't need to extend an existing class to create a PyTorch dataloader. All we need is an __iter__ method to retrieve batches of data, and an __len__ method to get the number of batches.

# In[ ]:


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# We can now wrap our data loaders using DeviceDataLoader.

# In[ ]:


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)


# Tensors that have been moved to the GPU's RAM have a device property which includes the word cuda. Let's verify this by looking at a batch of data from valid_dl.

# In[ ]:


for xb, yb in val_loader:
    print('xb.device:', xb.device)
    print('yb:', yb)
    break


# # Training the Model
# 
# We can use the exact same training loops from the logistic regression notebook.

# In[ ]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# We also define an accuracy function which calculates the overall accuracy of the model on an entire batch of outputs, so that we can use it as a metric in fit.
# 
# Before we train the model, we need to ensure that the data and the model's parameters (weights and biases) are on the same device (CPU or GPU). We can reuse the to_device function to move the model's parameters to the right device.

# In[ ]:


# Model (on GPU)
model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)
to_device(model, device)


# Let's see how the model performs on the validation set with the initial set of weights and biases.

# In[ ]:


history = [evaluate(model, val_loader)]
history


# The initial accuracy is around 10%, which is what one might expect from a randomly intialized model (since it has a 1 in 10 chance of getting a label right by guessing randomly).
# 
# We are now ready to train the model. Let's train for 5 epochs and look at the results. We can use a relatively higher learning of 0.5.

# In[ ]:


history += fit(5, 0.5, model, train_loader, val_loader)


# 96% is pretty good! Let's train the model for 5 more epochs at a lower learning rate of 0.1, to further improve the accuracy.

# In[ ]:


history += fit(5, 0.1, model, train_loader, val_loader)


# We can now plot the losses & accuracies to study how the model improves over time.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');


# In[ ]:


accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# Our current model outperforms the logistic regression model (which could only reach around 86% accuracy) by a huge margin! It quickly reaches an accuracy of 97%, but doesn't improve much beyond this. To improve the accuracy further, we need to make the model more powerful. As you can probably guess, this can be achieved by increasing the size of the hidden layer, or adding more hidden layers. I encourage you to try out both these approaches and see which one works better.

# # Summary and Further Reading
# 
# Here is a summary of the topics covered in this tutorial:
# 
# * We created a neural network with one hidden layer to improve upon the logistic regression model from the previous tutorial. We also used the ReLU activation function to introduce non-linearity into the model, allowing it to learn more complex relationships between the inputs (pixel densities) and outputs (class probabilities).\
# * We defined some utilities like get_default_device, to_device and DeviceDataLoader to leverage a GPU if available, by moving the input data and model parameters to the appropriate device.
# * We were able to use the exact same training loop: the fit function we had define earlier to train out model and evaluate it using the validation dataset.
# 
# There's a lot of scope to experiment here, and I encourage you to use the interactive nature of Jupyter to play around with the various parameters. Here are a few ideas:
# 
# * Try changing the size of the hidden layer, or add more hidden layers and see if you can achieve a higher accuracy.
# * Try changing the batch size and learning rate to see if you can achieve the same accuracy in fewer epochs.
# * Compare the training times on a CPU vs. GPU. Do you see a significant difference. How does it vary with the size of the dataset and the size of the model (no. of weights and parameters)?
# * Try building a model for a different dataset, such as the [CIFAR10 or CIFAR100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html).
# 
# Here are some references for further reading:
# 
# [A visual proof that neural networks can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html), also known as the Universal Approximation Theorem.
# 
# [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) - A visual and intuitive introduction to what neural networks are and what the intermediate layers represent
# 
# [Stanford CS229 Lecture notes on Backpropagation](http://cs229.stanford.edu/notes/cs229-notes-backprop.pdf) - for a more mathematical treatment of how gradients are calculated and weights are updated for neural networks with multiple layers.
