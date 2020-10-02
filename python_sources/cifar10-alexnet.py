#!/usr/bin/env python
# coding: utf-8

# # AlexNet on CIFAR10 
# The goal of this notebook is to reimplement [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  on a 32x32 pixel dataset called CIFAR10. AlexNet is a Convolutional Neural Network Architecture developed by Alex Krizhevsky, and published with Ilya Sutskever and Krizhevsky's doctoral advisor Geoffrey Hinton. It won the [ImagNet large scale Visual Recognition Challange](https://en.wikipedia.org/wiki/ImageNet#ImageNet_Challenge) in September 2012.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import time


# ### Transformations
# * I resize the images to 256 since AlexNet expects inputs of that size.
# * Perform Horizontal flips 50% of the time. This allows the model to recognize objects when they are facing the other direction too.
# * Make the images Tensors
# * Normalize the images. Now the stats I have below are the mean and standard deviations across all channels. I'll show below how I got them.
# 

# In[ ]:


transform= transforms.Compose(
    [   transforms.Resize(256),
     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
     transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223 , 0.24348513, 0.26158784])
    ])


# ### Train and Test set

# In[ ]:


trainset= torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform,
                                      download=True)
testset= torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform,
                                     download=True)


# In[ ]:


trainset.data.shape


# Below I get the mean and standard deviations of the pictures for each channel. These are the stats we use to normalize the images.

# In[ ]:


train_means= trainset.data.mean(axis=(0,1,2))/255
train_means


# In[ ]:


train_stds= trainset.data.std(axis=(0,1,2))/255
train_stds


# ### Dataloaders

# In[ ]:


trainloader= torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,
                                        num_workers=8)
testloader= torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False,
                                       num_workers=8)


# In[ ]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ### Get a validation set from the training

# In[ ]:


from torch.utils.data import Subset
def train_valid_split(dl, val_split=0.25):
    total_items= dl.dataset.data.shape[0]
    idxs= np.random.permutation(total_items)
    train_idxs, valid_idxs= idxs[round(total_items*val_split):], idxs[:round(total_items*val_split)]
    
    train= Subset(dl, train_idxs)
    valid= Subset(dl, valid_idxs)
    return train, valid


# In[ ]:


train_dl, valid_dl= train_valid_split(trainloader)


# ## Show images

# In[ ]:


import matplotlib.pyplot as plt
def show_image(img):
    img= img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    


# In[ ]:


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
show_image(torchvision.utils.make_grid(images[:4]))


# In[ ]:


[classes[each] for each in labels[:4]]


# ## Model
# The model I use is [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). It is the model which basically revolutionalized the space of Deep Learning and brought a lot of attention to this space of Image Classification.

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# trainloader.to(device);
print(device)


# In[ ]:


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        #1
        self.features= nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #2
        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #3
        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        #4
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        #5
        nn.Conv2d(384, 256, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool= nn.AvgPool2d(6)
        self.classifier= nn.Sequential(
            nn.Dropout(), nn.Linear(256*6*6, 4096), #128*2*2, 1024
        nn.ReLU(inplace=True), nn.Dropout(),
        nn.Linear(4096, num_classes))
        
    def forward(self, x):
        x= self.features(x)
        x=x.view(x.size(0), 256*6*6)
        x= self.classifier(x)
        return x


# Put the model on the GPU

# In[ ]:


model= AlexNet(num_classes=10).to(device)


# In[ ]:


model


# In[ ]:


#loss function and optimizer
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(params= model.parameters(), lr=3e-4)


# I create a function to convert seconds into a human friendly format of hours-minutes-seconds. This I will use to keep track of the training time.

# In[ ]:


import datetime

def convert_seconds_format(n):
    return str(datetime.timedelta(seconds =n))


# ### training loop
# Lets go for 10 epochs and see how the model performs.

# In[ ]:


all_losses=[]
all_valid_losses=[]
print('training starting...')
start_time= time.time()
for epoch in range(10):
    epoch_start=time.time()
    model.train()
    running_loss= 0.0
    running_valid_loss=0.0
    predictions=[]
    total=0
    correct=0
    
    for i, data in enumerate(train_dl.dataset, 0):

        inputs, labels= data[0].to(device), data[1].to(device)

        #zero parameter gradients
        optimizer.zero_grad()

        #forward + back optimize
        outputs= model(inputs)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #stats
        running_loss += loss.item()
    all_losses.append(running_loss/i)
    
    #evaluation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_dl.dataset, 0):
            inputs, labels= data[0].to(device), data[1].to(device)
            outputs= model(inputs)
            valid_loss= criterion(outputs, labels)
            running_valid_loss+= valid_loss.item()
            
            #the class with the highest score
            _, predicted= torch.max(outputs.data, 1)
            predictions.append(outputs)
            total+= labels.size(0)
            correct+= (predicted==labels).sum().item()
    epoch_end=time.time()
    epoch_time= convert_seconds_format(epoch_end-epoch_start)
    
    all_valid_losses.append(valid_loss)
    print(f"epoch {epoch+1}, running loss: {all_losses[-1]}")
    print(f"validation accuracy: {correct/total}. validation loss: {all_valid_losses[-1]}")
    print(f"epoch time: {epoch_time}")
end_time= time.time()
train_time= convert_seconds_format(end_time- start_time)
print('training complete')
print(f"total time to train: {train_time}")


# After 37 minutes of training, the moodel gets an accuracy of about 90% on the training set. Thanks Alex.

# In[ ]:



x_axis=[i for i in range(1, 11)]
x_axis


# In[ ]:


valid_losses_list=[each.item() for each in all_valid_losses]
    


# Below I plot the losses from the train and validation set over 10 epochs.

# In[ ]:


plt.plot(x_axis, all_losses, label='train')
plt.plot(x_axis, valid_losses_list, label='valid')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend();


# In[ ]:


correct, total=0, 0
predictions=[]


# In[ ]:


model.eval();
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels= data[0].to(device), data[1].to(device)
        #inputs= inputs.view(-1, 32*32*3)
        outputs= model(inputs)
        #the class with the highest score
        _, predicted= torch.max(outputs.data, 1)
        predictions.append(outputs)
        total+= labels.size(0)
        correct+= (predicted==labels).sum().item()


# In[ ]:


print(f' Accuracy score of: {correct/total}')


# On the testing data, the model is correct 82% of the time. This is really impressive of our model having built it from scratch. 
# 
# 
# 

# ### Conclusion
# After having reimplemented the AlexNet paper, I am glad it gets a great perfomance at classifying images on CIFAR10 datset. At the time of writing, the best model gets 96.53%, [as seen here](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130), which has a better architecture and tricks to improve model perfomance.
# 
# Please reach out to me if you spot something I might have missed as I am also still learning.
