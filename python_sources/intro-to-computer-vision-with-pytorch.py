#!/usr/bin/env python
# coding: utf-8

# # MNIST Image Classification using PyTorch Convolutional Neural Network

# In[ ]:


import numpy as np
import pandas as pd

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# # Data Acquisition
# 
# #### Load the training dataset into Pandas DataFrame

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# #### Split the training dataset to features and labels
# 

# In[ ]:


x_train_df = train.iloc[:,1:]
y_train_df = train.iloc[:,0]

print(x_train_df.shape, y_train_df.shape)


# #### Convert the data to numeric arrays and normalize the features

# In[ ]:


x_train = x_train_df.values/255.
y_train = y_train_df.values

x_test = test.values/255


# #### Reshape the test and training dataset to (28,28) image arrays

# In[ ]:


x_train = np.reshape(x_train, (-1, 1, 28,28))
x_test = np.reshape(x_test, (-1, 1, 28,28))


x_train.shape, x_test.shape


# #### Split the training dataset into training and validation datasets

# In[ ]:


# This is to ensure reproducibility
random_seed = 234
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)


x_train.shape, x_val.shape, y_train.shape, y_val.shape


# #### Helper function to display an array of images

# In[ ]:


def display(rows, columns, images, values=[], predictions=[]):
    fig = plt.figure(figsize=(9, 11))

    ax = []

    for i in range( columns*rows ):
        img = images[i]
        ax.append(fig.add_subplot(rows, columns, i+1))
        
        title = ""
        
        if(len(values) == 0):
            title = "Pred:" + str(predictions[i])
        elif(len(predictions) == 0):
            title = "Value:" + str(values[i])
        elif(len(values) != 0 and len(predictions) != 0):
            title = "Value:" + str(values[i]) + "\nPred:" + str(predictions[i])
        
        ax[-1].set_title(title)  # set title
        plt.imshow(img)

    plt.show()
    
idx = np.random.randint(1, 1000, size=9)

images = x_train[idx,:]
images = images[:,0]

values = y_train[idx]

display(rows=3, columns=3, images=images, values=values, predictions=[])


# #### Using the GPU

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# #### Lets define the Convolutional Neural Network Model

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1,2,3)
        
        self.fc1 = nn.Linear(338,10)
        self.fc1Size = 0
        self.toKnowMaxPoolSize= False
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        
        if(self.toKnowMaxPoolSize == True):
            self.fc1Size = x.size()
            print(x.size())
            return
        #now lets reshape the matrix i.e. unrolling the matrix
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        return x
    
net = Net()

net.to(device)

net


# #### Lets define the optimizer and loss function

# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# #### Define out tensors and dataloader

# In[ ]:


torch_x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)

train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = False)


# #### Train for 100 epoch

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n#Seed\ntorch.manual_seed(1234)\n\nfor epoch in range(10):\n    running_loss = 0.0\n    for i, data in enumerate(train_loader, 0):\n        inputs, labels = data\n        inputs, labels = inputs.to(device), labels.to(device)\n        \n        # zero the parameter gradients\n        optimizer.zero_grad()\n\n        # forward + backward + optimize\n        outputs = net(inputs)\n        outputs = F.log_softmax(outputs, dim=1)\n        loss = criterion(outputs, labels)\n        loss.backward()\n        optimizer.step()\n        \n        # print statistics\n        running_loss += loss.item()\n        if i % 1000 == 999:    # print every 500 mini-batches\n            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, loss.item()))\n#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / 500))\n#             running_loss = 0.0\n\nprint('Finished Training')")


# #### Try the model on the validation dataset

# In[ ]:


#Validate trained model
torch_x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
torch_y_val = torch.from_numpy(y_val).type(torch.LongTensor)

torch_x_val, torch_y_val = torch_x_val.to(device), torch_y_val.to(device)

val = net(torch_x_val)

_, predicted = torch.max(val.data, 1)

#Get accuration
print('Accuracy of the network %d %%' % (100 * torch.sum(torch_y_val==predicted) / len(y_val)))


# #### Lets display a sample of the predictions on the validation dataset

# In[ ]:


# Get random data from the valication dataset and the predicted values
idx = np.random.randint(1, 1000, size=9)

images = x_val[idx,:]
images = images[:,0]

values = y_val[idx]

predicted = predicted.cpu()

predictions = predicted.data.numpy()
predictions = predictions[idx]

display(rows=3, columns=3, images=images, values=values, predictions=predictions)


# #### Our model is awesome, lets use it on the test dataset

# In[ ]:


torch_x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

torch_x_test = torch_x_test.to(device)

y_test = net(torch_x_test)

_, predicted = torch.max(y_test.data, 1)


# #### Display predictions of the test dataset

# In[ ]:


idx = np.random.randint(1, 1000, size=9)

images = x_test[idx,:]
images = images[:,0]

predicted = predicted.cpu()

predictions = predicted.data.numpy()
predictions = predictions[idx]

display(rows=3, columns=3, images=images, values=[], predictions=predictions)


# #### Save predictions to csv for submission

# In[ ]:


ImageId = np.arange(1, len(x_test)+1)
Label = predicted.data.numpy()

my_submission = pd.DataFrame({'ImageId': ImageId, 'Label': Label})
my_submission.to_csv('submission.csv', index=False)

my_submission.head()

