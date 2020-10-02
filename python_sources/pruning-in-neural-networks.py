#!/usr/bin/env python
# coding: utf-8

# ###### In production, large neural networks can become a problem because they have quite a lot amount of weights. This can lead to increasing the size of the model as well as the time taken to predict the result. To overcome this problem, we use a method called pruning. There are a lot of methods of pruning a neural network. In this kernel, I basically implement two main methods of pruning:
# 
# ###### 1. Weight Pruning<br/> 2. Neuron Pruning

# In this kernel, i'll try to show how we can apply pruning techniques to neural networks using pytorch library

# ### Lets first import the libraries to create a neural network

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from pylab import rcParams
import copy
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# ### We will be using the MNIST data as it will be easier to explain using this data as it is not quite large in size

# In[ ]:


train_ds = datasets.MNIST(root='/tmp', train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.MNIST(root='/tmp', train=False, download=True, transform=transforms.ToTensor())


# In[ ]:


batch_size = 128


# In[ ]:


train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ## Creating a model 
# 
# ### Lets create a 4-layer fully connected neural network to classify digits of MNIST

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 200)
        self.fc5 = nn.Linear(200, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        out = self.fc5(x)
        return out


# In[ ]:


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# In[ ]:


num_epochs = 25


# ## Training the model
# 
#     ### The model is trained for 25 epochs and gets >96% accuracy

# In[ ]:


training_losses = []
training_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_dl)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _, predicted = torch.max(output.data, 1)
        total+=target.size(0)
        correct+=(predicted==target).sum().item()
    training_losses.append(running_loss/total)
    training_accuracies.append(correct/total)


# ### Here we have created plots for training loss and training accuracy of the model as the epoch increases

# In[ ]:


plt.figure(1, figsize=(15, 3))
plt.subplot(121)
plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.subplot(122)
plt.plot(training_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')


# In[ ]:


def test(model, test_dl):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_dl):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    return correct/total


# In[ ]:


original_accuracy = test(model, test_dl)
original_accuracy


# # Weight pruning
# 
# ## In weight pruning, we prune away k% of weights. 
# 
# For this, I have used the percentile function in numpy to get the kth smallest value in the weight. For all values in weight smaller than k, their weight will be set to zero. Also, we have to make sure that we don't affect the original model. For that purpose, I have used the copy module's deepcopy function to copy the model weights.
# 
# We don't have to prune the weights for the last output layer. That is why, I have used **i<length-2** because the *length-2* parameter corresponds to the weight for the last layer.

# In[ ]:


def weight_prune(model, pruning_percentage, test_dl):
    model1 = copy.deepcopy(model)
    length = len(list(model1.parameters()))
    for i, param in enumerate(model1.parameters()):
        if len(param.size())!=1 and i<length-2:
            weight = param.detach().cpu().numpy()
            weight[np.abs(weight)<np.percentile(np.abs(weight), pruning_percentage)] = 0
            weight = torch.from_numpy(weight).to(device)
            param.data = weight
    return test(model1, test_dl)


# In[ ]:


pruning_percent = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]


# In[ ]:


accuracy_weight_pruning = []
for percent in pruning_percent:
    accuracy_weight_pruning.append(weight_prune(model, percent, test_dl))


# ### Plotting the original accuracy in red and the accuracies for the weight-pruned accuracies in blue

# In[ ]:


rcParams['figure.figsize'] = 12, 8
plt.plot(pruning_percent, np.size(pruning_percent)*[original_accuracy], 'r',
         pruning_percent, accuracy_weight_pruning, 'b')
plt.grid()
plt.legend([['Original Accuracy'], 
            ['Accuracy with weight pruning']],
           loc='lower left', fontsize='xx-large')
plt.xlabel('Pruning Percentage', fontsize='xx-large')
plt.ylabel('Accuracy', fontsize='xx-large')
plt.xticks(pruning_percent)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.show()


# # Neuron Pruning
# 
# ## In neuron pruning, we rank the columns of the weight by their l2-norm and the smallest k% are deleted.
# 
# For this purpose, I used the numpy's linalg.norm() method, which by default gives the l2-norm and set the columns where the norm is less than the pruning percentage to zero. Again I use copy module's deepcopy function to create a new sparse model.
# 
# We don't have to prune the weights for the last output layer. That is why, I have used **i<length-2** because the *length-2* parameter corresponds to the weight for the last layer.

# In[ ]:


def neuron_pruning(model, pruning_percentage, test_dl):
    model1 = copy.deepcopy(model)
    length = len(list(model1.parameters()))
    for i, param in enumerate(model1.parameters()):
        if len(param.size())!=1 and i<length-2:
            weight = param.detach().cpu().numpy()
            norm = np.linalg.norm(weight, axis=0)
            weight[:, np.argwhere(norm<np.percentile(norm, pruning_percentage))] = 0
            weight = torch.from_numpy(weight).to(device)
            param.data = weight
    return test(model1, test_dl)


# In[ ]:


accuracy_neuron_pruning = []
for percent in pruning_percent:
    accuracy_neuron_pruning.append(neuron_pruning(model, percent, test_dl))


# ## Plotting the accuracy achieved due to neuron pruning

# In[ ]:


rcParams['figure.figsize'] = 12, 8
plt.plot(pruning_percent, np.size(pruning_percent)*[original_accuracy], 'r', pruning_percent, accuracy_neuron_pruning, 'b')
plt.grid()
plt.legend([['Original Accuracy'], ['Accuracy with neuron pruning']], loc='lower left', fontsize='xx-large')
plt.xlabel('Pruning Percentage', fontsize='xx-large')
plt.ylabel('Accuracy', fontsize='xx-large')
plt.xticks(pruning_percent)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.show()


# # Comparing weight pruning and neuron pruning by plotting them on same plot.

# In[ ]:


rcParams['figure.figsize'] = 12, 8
plt.plot(pruning_percent, np.size(pruning_percent)*[original_accuracy], 'r',
         pruning_percent, accuracy_weight_pruning, 'b',
         pruning_percent, accuracy_neuron_pruning, 'g')
plt.grid()
plt.legend([['Original Accuracy'],
            ['Accuracy with weight pruning'], 
            ['Accuracy with neuron pruning']], 
           loc='lower left', fontsize='xx-large')
plt.xlabel('Pruning Percentage', fontsize='xx-large')
plt.ylabel('Accuracy', fontsize='xx-large')
plt.xticks(pruning_percent)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.show()


# ## Conclusion
# 
# This pruning method can be applied to any neural networks to decrease their size and time it takes to predict the answer for the test set. There are more advanced pruning techniques to apply to neural networks. I encourage the reader to search for these techniques on the internet and apply on his project to make his production time fast.
