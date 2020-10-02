#!/usr/bin/env python
# coding: utf-8

# # Foreword
# This tutorial is part of a series of tutorials on PyTorch, the deep learning library developed by Facebook. This is the second tutorial of the series. To go to the first tutorial please go to this notebook: https://www.kaggle.com/krishanudb/pytorch-tutorial-for-beginners
# 
# In this tutorial we will learn how to make simple computation graphs like linear regression and logistic regression using PyTorch. So lets jumo right in.
# 
# ### Acknowledgements
# https://github.com/yunjey/pytorch-tutorial

# # Linear Regression

# #### Importing the packages

# In[ ]:


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# #### Defining the Hyperparameters

# In[ ]:


input_size = 1
output_size = 1
num_epochs = 10000
learning_rate = 0.001


# #### Defining a Toy Dataset

# In[ ]:


x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],[9.779], [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# #### Linear Regression Model

# In[ ]:


model = nn.Linear(input_size, output_size)

# Loss Function:
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# #### Training the Model

# In[ ]:


for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print("Epoch: {}/{}; \tLoss: {}".format(epoch + 1, num_epochs, loss.item()))


# #### Plotting the outputs

# In[ ]:


predicted = model(torch.from_numpy(x_train)).detach().numpy()

plt.scatter(x_train, y_train, label='original data')
plt.scatter(x_train, predicted, label='predicted data')
plt.legend()
plt.show()


# # Logistic Regression Model
# We will use a logistic regression model to classify the MNIST dataset

# In[ ]:


import torchvision
import torchvision.transforms as transforms


# #### Defining Hyperparameters

# In[ ]:


input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001


# #### loding the Dataset

# In[ ]:


train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train = False, transform=transforms.ToTensor(), download=True)


# In[ ]:


plt.imshow(train_dataset.train_data[0])
plt.show()


# #### Create DataLoader objects

# In[ ]:


trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


model = nn.Linear(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# #### Training the Model

# In[ ]:


total_step = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        if (i + 1) % 200 == 0:
            print("Epoch: {}/{}, \tIteration: {}/{}, \tLoss: {}".format(epoch + 1, num_epochs, i + 1, len(trainloader), loss.item()))


# #### Testing the model
# In this phase we donot need to keep track of the gradients. So in order to save memory, we use the torch.no_grad() function

# In[ ]:


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        total += labels.size(0)
    print("Accuracy of the model: {}".format(float(correct) / float(total)))
        
        


# # Feed Forward Neural Network Model

# #### Defining the network parameters

# In[ ]:


input_size = 28 * 28
hidden_size = 500
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001


# #### Create DataLoader Objects

# In[ ]:


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = True)


# #### Device Configuration

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### Model definition

# In[ ]:


import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out    


# In[ ]:


model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# #### Training the Model

# In[ ]:


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 200 == 0:
            print("Epoch: {}/{};\tIteration: {}/{}; Loss: {}".format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))


# In[ ]:


with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        total += labels.size(0)
print("Accuracy of the model {}".format(float(correct)/float(total)))


# ## The neural network model performs slightly better than the Logistic Regression Model

# That brings us to the end of the second tutorial of the series. Hope you liked it.
# The next tutorial of the series will introduce Convolutional Neural Networks and gradually move on to more advanced model architectures like inception networks and UNets. Do check out the tutorial at: https://www.kaggle.com/krishanudb/tutorial-on-convolutional-nets-in-pytorch

# In[ ]:




