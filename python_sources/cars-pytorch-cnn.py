#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os, math, random, time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image

import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

from collections import defaultdict


# In[ ]:


def timer_start():
    global t0
    t0 = time.time()
    
def timer_end():   
    print('Time elapsed {:0.1f}s'.format(time.time() - t0))  

def display_grid(data, path, w =10, h =10, columns = 4, rows = 5):
    fig=plt.figure(figsize=(12, 8))
    for i in range(1, columns*rows +1):
        file = data[i]
        file = os.path.join(path, file)
        img = Image.open(file)
        fig.add_subplot(rows, columns, i)
        imshow(img)
    plt.show()
    
def get_best_epcoh(history):
    valid_acc = history['val_acc']
    best_epoch = valid_acc.index(max(valid_acc)) +1
    best_acc =  max(valid_acc)
    print('Best Validation Accuracy Score {:0.5f}, is for epoch {}'.format( best_acc, best_epoch))
    return best_epoch

def plot_results(history):
    tr_acc = history['tr_acc']
    val_acc = history['val_acc']
    tr_loss = history['tr_loss']
    val_loss = history['val_loss']
    epochs = history['epoch']

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, tr_acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')  
    
    plt.subplot(1,2,2)
    plt.plot(epochs, tr_loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()
    
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


# ## Class Swift

# In[ ]:


base_dir = '/kaggle/input/cars-wagonr-swift/data/'
train_swift = os.listdir(os.path.join(base_dir, 'train/swift') )
val_swift  = os.listdir(os.path.join(base_dir, 'validation/swift') )
test_swift  =  os.listdir(os.path.join(base_dir, 'test/swift') )
print('Instances for Class Swift: Train {}, Validation {} Test {}'.format(len(train_swift), len(val_swift), len(test_swift)))


# In[ ]:


#Sanity checks: no overlaping bteween train test and validation sets
val_train = [x for x in val_swift if x in train_swift]
test_train = [x for x in test_swift if x in train_swift]
val_test =  [x for x in test_swift if x in val_swift]
len(val_train), len(test_train), len(val_test)


# In[ ]:


display_grid(data = train_swift, path = os.path.join(base_dir, 'train/swift'), w =10, h =10, columns = 8, rows = 5)


# ## Class Wagonr

# In[ ]:


train_wr = os.listdir(os.path.join(base_dir, 'train/wagonr') )
val_wr  = os.listdir(os.path.join(base_dir, 'validation/wagonr') )
test_wr  =  os.listdir(os.path.join(base_dir, 'test/wagonr') )
print('Instances for Class Wagonr: Train {}, Validation {} Test {}'.format(len(train_swift), len(val_swift), len(test_swift)))


# In[ ]:


#Sanity checks: no overlaping bteween train test and validation sets
val_train = [x for x in val_wr if x in train_wr]
test_train = [x for x in test_wr if x in train_wr]
val_test =  [x for x in test_wr if x in val_wr]
len(val_train), len(test_train), len(val_test)


# In[ ]:


display_grid(data = train_wr, path = os.path.join(base_dir, 'train/wagonr'), w =10, h =10, columns = 8, rows = 5)


# ## Data Preprocessing

# In[ ]:


# from https://forums.fast.ai/t/normalizing-your-dataset/49799
# Compute the mean and standrad deviation of the training images for each channel. This will be used to normalize the tensors to [-1,1]
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation' )

transform = transforms.Compose( [                                  
                                 transforms.Resize((150,150)), 
                                 transforms.ToTensor(),
                                  ])

trainset = torchvision.datasets.ImageFolder( root= train_dir ,
                                              transform=transform
                                               )

trainloader = torch.utils.data.DataLoader(trainset, batch_size= 512 ,
                                          shuffle=True, num_workers=1)
mean, std = online_mean_and_sd(trainloader)
print(mean, std)


# In[ ]:




BATCH_SIZE = 20
# transforms.ToTensor() trasnforms the pixels from [0,255] to [0,1] which is then 
# trasnformed to [-1,1] using Normalize with mean computed mean and std for each of three channels
transform = transforms.Compose( [                                  
                                 transforms.Resize((150,150)), 
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std),
                                  ])

trainset = torchvision.datasets.ImageFolder( root= train_dir ,
                                              transform=transform
                                               )


trainloader = torch.utils.data.DataLoader(trainset, batch_size= BATCH_SIZE,
                                          shuffle=True, num_workers=1)


validset = torchvision.datasets.ImageFolder( root= validation_dir ,
                                              transform=transform
                                               )


validloader = torch.utils.data.DataLoader(validset , batch_size= BATCH_SIZE,
                                          shuffle=True, num_workers=1)
#Verify that mean is 0 and SD = 1
print(online_mean_and_sd(trainloader))


# ## Build CNN model

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=32, kernel_size= 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride= 2)
        
        self.conv2 =  nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3)
        self.conv3 =  nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3)
        self.conv4 =  nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3)
    
#       128 * 128 * 7 is the output of the last max pool layer
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 2)
       

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        #this is similar to flatten in keras but keras is smart to figure out dimensions by iteself.
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x


# ## Train Model

# ### Custom Train and Test Function

# In[ ]:



def train(model, epoch, train_loader, optimizer, criterion) :   
    model.train()
    epoch_loss = correct = 0
    for i, data in enumerate(train_loader, 0):
        # Load images with gradient accumulation capabilities
        inputs, labels = data[0].to(device), data[1].to(device)
        # Clear gradients w.r.t. parameters       
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(inputs)

         # Calculate Loss: softmax --> cross entropy loss
        loss =  criterion(outputs, labels)
        
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)           

        # Total correct predictions 
        correct += (predicted == labels).sum().item()  

        # Getting gradients w.r.t. parameters
        loss.backward()

         # Updating parameters
        optimizer.step() 
        
        #Multiple loss by number of batch as loss is averaged per batch
        epoch_loss += outputs.shape[0] * loss.item()
        
    accuracy = correct / len(train_loader.dataset)
    epoch_loss = epoch_loss / len(train_loader.dataset)
    return epoch_loss, accuracy


def test(model, epoch, test_loader, optimizer, criterion):
    model.eval()
    epoch_loss = correct = 0

    with torch.no_grad():
        #Iterate through test dataset after every epoch
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # Forward pass only to get logits/output
            outputs = model(images)

            # Calculate Validation Loss
            loss =    criterion(outputs, labels)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)           

            # Total correct predictions 
            correct += (predicted == labels).sum().item()  

             #Multiple loss by number of batch as loss is averaged per batch so that we get
            #total loss over an epoch and then divide by number of samples to get loss per epcoh
            epoch_loss += outputs.shape[0] * loss.item()

    
    accuracy = correct / len(test_loader.dataset)
    epoch_loss = epoch_loss / len(test_loader.dataset)
  
    return epoch_loss, accuracy 



# ### Set Loss Function and Optimizers

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
print('Divice: ',device)
model


# ### Train and Validate

# In[ ]:


get_ipython().run_cell_magic('time', '', "EPOCHS = 50\n\n#To get reprodicible results but not working at mommnet\n# set_seed(42)\n\ntrain_list  = os.listdir(os.path.join(base_dir, 'train/swift') ) + os.listdir(os.path.join(base_dir, 'train/wagonr') )\nnum_batches = math.ceil(len(train_list) / BATCH_SIZE)\n\nprint('Number of Training samples {}, Batch Size {}, Num Batch {}'.format( len(train_list), BATCH_SIZE, num_batches ))\n\nhistory = defaultdict(list)\n\n\n# Get Keras like outputs for Training and validation by using custom train and test functions.\nfor epoch in range( EPOCHS):  \n    timer_start()\n    print('[Epoch {} of {}]'.format(epoch +1, EPOCHS), end = ' ')\n    tr_loss, tr_acc = train(model, epoch, trainloader, optimizer, criterion)\n\n    val_loss, val_acc = test(model, epoch, validloader, optimizer, criterion)  \n    timer_end()\n    print('tr_loss: {:0.4f},tr_acc {:0.4f}| val_loss {:0.4f}, val_acc {:0.4f}'.format(tr_loss, tr_acc , val_loss, val_acc))\n    history['epoch'].append(epoch+1)\n    history['tr_loss'].append(round(tr_loss,5))\n    history['tr_acc'].append(round(tr_acc,5))\n    history['val_loss'].append(round(val_loss,5))\n    history['val_acc'].append(round(val_acc,5))")


# ## Plot Training vs Validation results[](http://)

# In[ ]:



plot_results(history)
best_epoch = get_best_epcoh(history)

