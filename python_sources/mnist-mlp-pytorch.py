#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')


# # Steps to create a Neural Networks with Pytorch
# 
# 1. Define a Neural Network (forward propagation)
# 2. Define dataset, data loader and transformations
# 3. Define optimizer and criterion
# 4. Train the model (back propagation)
# 5. Evaluate the model
# 6. Make prediction

# # Step 1: Define a Neural Networks
# The neural network architectures in Pytorch can be defined in a class which inherits the properties from the base class from **nn** package called Module. This inheritance from the nn.Module class allows us to implement, access, and call a number of methods easily. We can define all the layers inside the constructor of the class, and the forward propagation steps inside the forward function.
# 
# We will define a simple Multilayer Perceptron with the following architecture:
# 
# * Input layer
# ```Python
# nn.Linear(28 * 28, 512)
# ```
#     * Layer type: nn.Linear(), which refers to a fully connection layer
#     * Input size: 28*28, corresponding to the size of input data.
#     * Output size: 512, the number of "neurons".
#     
# * Hidden layer
# ```
# nn.Linear(512, 256)
# ```
# 
#     * Layer type: nn.Linear()
#     * Input size: 512, output size of the previous layer(input layer).
#     * Output size: 256, the number of "neurons" in this layer.
#     
# * Output layer
# ```
# nn.Linear(256, 10)
# ```
# 
#     * Layer type: nn.Linear()
#     * Input size: 256, output size of the previous layer(hidden layer).
#     * Output size: 10, the number of classes we need to predict.
# 
# * Activation functions
# Each linear layer's output needs to go through an activation function to "activate" it. We will get started with **F.sigmoid()** but can try F.relu() or others later.
# 
# The best practice is to name each layer and initialize them in the **__init__()** function as named building blocks and put the building blocks together in the **forward()** function which defines how the data actually flows in the network. In our case, each layer simply takes the output of the previous layer and perform transformations the generate outputs in sequence.

# In[ ]:


class Net(nn.Module):
## The following two lines are the reinforced format for contructing a Pytorch network class
    def __init__(self):
        super(Net, self).__init__()
# Input layer
        self.input = nn.Linear(28 * 28, 512)
# Hidden layer
        self.hidden = nn.Linear(512, 256)
# Output layer
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.sigmoid(self.input(x))
        x = F.sigmoid(self.hidden(x))
        x = self.output(x)
        return x
    
model = Net()
print(model)


# # Step 2: Define dataset, data loader and transformation
# 
# PyTorch gives use the freedom to pretty much do anything with the Dataset class so long as you override two of the subclass functions:
# * the __len__ function which returns the size of the dataset, and
# * the __getitem__ function which returns a sample from the dataset given an index.
# 
# However, in our case we can simply construct a TensorDataset with two items: the feature data and the target where the feature data is the matrix of pixel 1 - pixel 784 and the target is the digit of the image.
# 
# 
# While the Dataset class is a nice way of containing data systematically, it seems that in a training loop, we will need to index or slice the dataset's samples list. This is no better than what we would do for a typical list or NumPy matrix. Rather than going down that route, PyTorch supplies another utility function called the DataLoader which acts as a data feeder for a Dataset object.
# 
# 
# In order to construct the data loader we will need to provide two parameters: **batch_size** which indicates how many samples we want to use to train  the model in a batch, and **shuflle**, suggesting if we want to shuffle the data before sending it to the network. 
# 
# Typically we would want to set batch_size as 2^N e.g. 128, 256, 512, and set shuffle as True for traning data and False for validatio and test data( you can take a moment to think why?)
# 
# 

# In[ ]:


batch_size = 128
transform = transforms.ToTensor()


x_train, x_val, y_train, y_val = train_test_split(
    train.values[:,1:], train.values[:,0], test_size=0.2)


train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train.astype(np.float32)/255),
                                               torch.from_numpy(y_train))

val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_val.astype(np.float32)/255),
                                               torch.from_numpy(y_val))

test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test.values[:,:].astype(np.float32)/255))

# data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


# # Step 3: Define optimizer and criterion
# 
# Optimizer is used to perform the gradient descent process. Here we will use SGD( Stochastic Gradient Descent). The tricky part is how to set the right size of learning rate which could have a huge impact on the final result. For now let's simply use 0.1 as the starting point but later we will revisit the options and strategies of learning rate selection.
# 
# Criterion will be used to calculate the cost (or loss) so we can use the cost to do back propagation and update the weights we want to train. In our case, we will use nn.CrossEntropyLoss() since we are working on a multiclassfication problem.

# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# # Step 4 train the model (back propagation)
# 
# Training the model is an iterative process which contains many epoches. For each epoch we will repeatly load batches of data, perform forward propagation, calculate cost, perform back propagation using the optimizer.

# In[ ]:


n_epochs = 30
for epoch in range(n_epochs):
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        # Forward propagation
        output = model(data)
        # Calculate the loss
        loss = criterion(output, target)
        # Back propagation
        loss.backward()
        # Update weights using the optimizer
        optimizer.step()
        # Calculate the cumulated loss
        train_loss += loss.item()*data.size(0)
    
    train_loss = train_loss/len(train_loader.dataset)
    
    print(f"Epoch: {epoch}, train loss: {train_loss}")


# # Step 5: Evaluate the model
# 
# We will use the trained model to make predictions on the validation dataset and compare the predictions against the actual targets. Dataloader will be used to iterate the validation dataset as well.

# In[ ]:


val_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for evaluation

for data, target in val_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update val loss 
    val_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate val accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg val loss
val_loss = val_loss/len(val_loader.sampler)
print('val Loss: {:.6f}\n'.format(val_loss))

for i in range(10):
    if class_total[i] > 0:
        print('val Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('val Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nval Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


# # Step 6: make prediction
# 
# This step is similar to the validation step except that we are not comparing the predictions as there's no ground truth of target to compare with.

# In[ ]:


model.eval() # prep model for evaluation

preds = []

for data in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data[0])
    # calculate the loss
    _, pred = torch.max(output, 1)
    preds.extend(pred.tolist())
    # compare predictions to true label


# In[ ]:


submission['Label'] = preds
submission.to_csv('submission.csv', index=False)


# # Assignments:
# 
# 1. Can you implment the following functions for model training, evaluation and prediction so we can reuse them when we need to test different things afterwards without having to replicate the codes every time.
# 
# ```python
# 
# def train_model(nn_model, train_loader, optimizer, criterion, n_epoch):
#     # YOUR IMPLEMENTATION
#     return nn_model
# 
# def eval_model(nn_model, val_loader):
#     # YOUR IMPLEMENTATION
#     return calculated_accuracy
# 
# 
# def predict(nn_model, test_loader):
#     # YOUR IMPLEMENTATION
#     return predictions
# 
# ```
# 
# 
# 2. F.sigmoid() was used as the activation function for the MLP model we implemented. Can you try other activation functions such as F.relu() and F.tanh()? You may want to refer to the [PyTorch Functional](https://pytorch.org/docs/stable/nn.functional.html) for more details. Use the train_model(), eval_model() functions you implmented so you don't have to repeat the same codes.
# 
# 3. Can you add a dropout layer between input and the hidden layer, and another one between the hidden layer and the output layer?
# 
# 4. Try to add/ remove hiden layers, as well as different number of neurons and report your validation results.
# 
# 5. Try different learning rate (0.0001, 0.001, 0.01, 0.1). What is your observation?
# 
# 6. Try different values for batch_size(64, 128, 256, 512).
# 
# 7. Try different values for n_epochs.
# 
# 8. What can you think of if we want to improve the current model?
# 
# 

# In[ ]:




