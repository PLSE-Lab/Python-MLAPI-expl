#!/usr/bin/env python
# coding: utf-8

# ### Welcome to the Part-3 of PyTorch Series. <br>
# ## Logistic Regression using PyTorch

# Link to the Previous Notebook -> https://www.kaggle.com/superficiallybot/getting-started-with-pytorch-series-part-2

# In the upcoming post, I'll discuss **Feed Forward Neural Networks**. Stay tuned!

# Notebook Details:-<br>
# 1. Framework Used -> PyTorch
# 2. Dataset Used -> MNIST

# In[ ]:


# importing the libraries

import torch
# Variable object automatically does require_autograd = True
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets


# Steps
#  
# 1. Loading and Preparation of Dataset
# 2. Making the dataset iterable
# 3. Create Model Class
# 4. Instantiate Model Class
# 5. Instantiate Loss Class
# 6. Instantiate Optimizer Class
# 7. Train Model

# ### Step 1 -> Loading and Preparing the Dataset

# In this step, you load the dataset and do preparations on it (transformations) so that it can be used by the Deep Learning Models.<br>
# In your case, we are using MNIST dataset which is provided by the PyTorch library and it is preprocessed. We need not to make much changes. <br>
# However, in most of the cases, you would require to implement the Dataset class describing how your dataset would be handled.

# In[ ]:


# torchvision.datasets has preloaded common datasets, ready to use. 
train_dataset = dsets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = True)


# In[ ]:


# Hyper-parameters settings
batch_size = 100
n_iters = 3000
epochs = n_iters / (len(train_dataset)/ batch_size)

input_dim = 784
output_dim = 10
lr_rate = 0.001


# ### Step 2 -> Making the dataset iterable using DataLoader

# In[ ]:


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)


# ### Step 3 -> Creating the Deep Learning Model Class

# Here you<br>
# Class name can be anything. <br>
# It can be logistic regression, it can be 'A', can be 'xyz' etc.

# In[ ]:


class a(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(a, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs


# ### Step 4 -> Instantiating the Model Class

# In[ ]:


model = a(input_dim, output_dim)


# ### Step 5 -> Instantiating the Loss Class

# In[ ]:


# loss Class 
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy


# ### Step 6 -> Instantiate the Optimizer Class

# In[ ]:


optimizer = torch.optim.SGD(model.parameters(), lr = lr_rate)


# ### Step 7 -> Training the Model

# Since the model isn't computationally expensive, you can run the notebook in 'cpu' device mode. **However, for Neural Networks you would require to switch to 'gpu' mode.**

# In[ ]:


# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' %(epoch + 1, num_epochs, i+ 1, len(train_dataset)//batch_size, loss.item()))
    


# ### Evaluating the Model

# In[ ]:


# Test the model
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    
    correct += (predicted == labels).sum()
    
print('Accuracy of the model on the 10.000 test images: %d ' %(100 * correct / total))
    

