#!/usr/bin/env python
# coding: utf-8

# This is my first attempt at a Kaggle Kernel. I am currently learning how to use pytorch as well. In this notebook, we will take a look at the Fruit-360 data and try to fit a neural network to classify the fruit images. I noticed that this data-set gets additional images and categories periodically, so I designed the network so that it will be able to make a proper fit regardless if the data-set has been expanded. Here we go! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_files
import torch.optim as optim
import os
import numpy as np
import time
from PIL import Image
from torchvision.utils import make_grid
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from glob import glob


# In[ ]:


path = '../input/fruits-360_dataset/fruits-360'


# First, I want to check how many images there are in the subfolder 'Training':

# In[ ]:


files_training = glob(os.path.join(path,'Training', '*/*.jpg'))
num_images = len(files_training)
print('Number of images in Training file:', num_images)


# This is a pretty decent number of images, so I think that we have enough to split these images into a 'train' and 'valid' dataset. First, lets also take a look at how many different classes of fruit there are, and how many images there are of each different type of fruit.

# In[ ]:


min_images = 1000
im_cnt = []
class_names = []
print('{:18s}'.format('class'), end='')
print('Count:')
print('-' * 24)
for folder in os.listdir(os.path.join(path, 'Training')):
    folder_num = len(os.listdir(os.path.join(path,'Training',folder)))
    im_cnt.append(folder_num)
    class_names.append(folder)
    print('{:20s}'.format(folder), end=' ')
    print(folder_num)
    if (folder_num < min_images):
        min_images = folder_num
        folder_name = folder
        
num_classes = len(class_names)
print("\nMinumum imgages per category:", min_images, 'Category:', folder)    
print('Average number of Images per Category: {:.0f}'.format(np.array(im_cnt).mean()))
print('Total number of classes: {}'.format(num_classes))


# Now real quick I will try to get an estimate of the mean of each channel from the images and an estimate of the standard deviation of the channels so I can normalize the imputs:

# In[ ]:


tensor_transform = transforms.Compose([
    transforms.ToTensor()
])

all_data = ImageFolder(os.path.join(path, 'Training'), tensor_transform)


# In[ ]:


data_loader = torch.utils.data.DataLoader(all_data, batch_size=512, shuffle=True)


# In[ ]:


pop_mean = []
pop_std = []

for i, data in enumerate(data_loader, 0):
    numpy_image = data[0].numpy()
    
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std = np.std(numpy_image, axis=(0,2,3))
    
    pop_mean.append(batch_mean)
    pop_std.append(batch_std)

pop_mean = np.array(pop_mean).mean(axis=0)
pop_std = np.array(pop_std).mean(axis=0)


# In[ ]:


print(pop_mean)
print(pop_std)


# So there are 95 different classes of fruit, and there are on average 515 images for each category. It looks like most categories have about the same number of fruit, so I will not worry too much about not having enough pictures in any given category. Now I will create a shuffle that will be used to split the 'Training' data into a 'train' and 'valid' set:

# In[ ]:


np.random.seed(123)
shuffle = np.random.permutation(num_images)


# In[ ]:


split_val = int(num_images * 0.2)
print('Total number of images:', num_images)
print('Number of valid images after split:',len(shuffle[:split_val]))
print('Number of train images after split:',len(shuffle[split_val:]))


# Now I'm going to create some sub-classes that extends the Dataset class so I can read the data in batches. I realize now that there was probably a more efficient way of doing this, but this will get the job done. I created a subclass for loading in the train_data, valid_data, and test_data:

# In[ ]:


class FruitTrainDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[split_val:]])
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y

class FruitValidDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[:split_val]])
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y
    
class FruitTestDataset(Dataset):
    def __init__(self, path, class_names, transform=transforms.ToTensor()):
        self.class_names = class_names
        self.data = np.array(glob(os.path.join(path, '*/*.jpg')))
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y
    


# Now I will define some data transformations and create the three different datasets. 

# In[ ]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ]),
    'Test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ])
}

train_dataset = FruitTrainDataset(files_training, shuffle, split_val, class_names, data_transforms['train'])
valid_dataset = FruitValidDataset(files_training, shuffle, split_val, class_names, data_transforms['valid'])
test_dataset = FruitTestDataset('../input/fruits-360_dataset/fruits-360/Test', class_names, transform=data_transforms['Test'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# In[ ]:


### These are just used for convinience purposes

dataloaders = {'train': train_loader,
              'valid': valid_loader,
              'Test': test_loader}
dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(valid_dataset),
    'Test': len(test_dataset)
}


# In[ ]:


# This allows us to see some of the fruits in each of the datasets 
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = pop_std * inp + pop_mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# In[ ]:


# Here we are just checking out the next batch of images from the train_loader, and below I print the class names. 
inputs, classes = next(iter(train_loader))
out = make_grid(inputs)

cats = ['' for x in range(len(classes))]
for i in range(len(classes)):
    cats[i] = class_names[classes[i].item()]
    
imshow(out)
print(cats)


# ## Defining the Network

# Okay, now on to the fun part. Below I create a class Net from scratch. The architecture of course wasn't my first attempt, but it follows the basic pattern that we typically be most successful with CNN, namely several convolutional-pooled layers followed by one or two (two in this case) fully connected layers. 

# In[ ]:


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 10 * 10, 250)
        self.fc2 = nn.Linear(250, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(-1, 64 * 10 * 10)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training, p=0.4)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1) 


# We now create an instance of the network and run it on GPU. We also define an optimizer, loss criterion, and a scheduler. 

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


model = Net(num_classes)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
exp_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# We now need to be able to train our model, which can be done with the function bellow:

# In[ ]:


def fit(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time() # allows us to keep track of how long it took
    
    best_acc = 0.0 # allows us to store the best_acc rate (for validation stage)
    
    # Loop through the data-set num_epochs times.
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 15)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train() # This sets the model to training mode
            else:
                model.eval() # this sets the model to evaluation mode 
                
            running_loss = 0.0
            running_corrects = 0
            
            # using the dataloaders to load data in batches
            for inputs, labels in dataloaders[phase]:
                # putting the inputs and labels on cuda (gpu)
                inputs = inputs.to(device) 
                labels = labels.to(device)
                
                # zero the gradient
                optimizer.zero_grad()
                
                # if training phase, allow calculating the gradient, but don't allow otherwise
                with torch.set_grad_enabled(phase == 'train'):
                    # get outputs and predictions
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels) # get value of loss function with the current weights 
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # keep track of the best weights for the validation dataset 
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model


# We will now apply the function to train our model 

# In[ ]:


model = fit(model, criterion, optimizer, exp_scheduler, num_epochs=30)


# # Results:

# In[ ]:


correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloaders['Test']:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {} test images: {:.3f}%'.format(dataset_sizes['Test'], 
                                                                      100 * correct / dataset_sizes['Test']))

