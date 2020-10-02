#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In this competition, we are given a dataset of handwritten images . The handwritten images are of digits from 0 to 9. We have to then submit the labels file for the new test images.
# We will be creating a convolutional neural network and then train it for 20 epochs. I will also do some data augmentation, and also apply learning rate annealing. 

# # Loading the required libraries. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler


# # Reading and Viewing the input data files. 

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


print("Number of rows in the traning data are : ", df_train.shape[0])
print("Number of rows in the test data are : ", df_test.shape[0])


# Viewing some rows of the training data and test data

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# So, we can see that image pixels are given in the form of a 1-D array for each image of dimension 784. Also train dataset has label column which gives us the label for the image.  

# In[ ]:


y_train = df_train['label']
x_train = df_train.drop('label', axis=1)
x_train = x_train.as_matrix()
print("Shape of the training data is -:", x_train.shape)


# In[ ]:


fig = plt.figure(figsize=(8, 8))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    img = np.array(x_train[i].reshape(28, 28))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap='gray')
plt.show()


# so, we can see that the images are of number and are grayscale.

# # Creating custom Data Loaders for the MNIST dataset 

# We need to create a custom data loader in pytorch for loading the dataset. Our custom dataset should inherit Dataset and override the methods : 
#  __len__ so that len(dataset) returns the size of the dataset.
#  __getitem__ to support the indexing such that dataset[i] can be used to get i
# We will read the csv in __init__ method
# 
# More about this can be read from-:
# 1) https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# 2) https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# In[ ]:


class CustomDatasetFromCSV_new(Dataset):
    
    # In this we read the dataset csv file and then converting the training data file into train and label file. 
    def __init__(self, csv_path, transforms=None, is_test=False):
        self.data = pd.read_csv(csv_path)
        self.is_test = is_test
        if not self.is_test:
            self.labels = np.asarray(self.data.iloc[:, 0])
            self.data = self.data.iloc[:, 1:]
        # Declaring torchvision transforms. more about these can be read here https://pytorch.org/docs/0.2.0/torchvision/transforms.html
        self.transforms = transforms
        
    
    # This function returns an image and label from the dataset.
    def __getitem__(self, index):
        
        # Reshaping the image to size 28*28
        img_as_np = np.array(np.reshape(self.data.iloc[index], (28, 28))).astype(np.uint8)
        img_as_img = Image.fromarray(img_as_np)        
        #Converting the image to black and white form. 
        img_as_img = img_as_img.convert('L')
        # Applying torchvision transforms.
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        
        if not self.is_test:
            single_image_label = self.labels[index]
            # Return image and the label
            return (img_as_tensor, single_image_label)
        return img_as_tensor
        
    # This returns the length of the dataset
    def __len__(self):
        return len(self.data.index)


# # Convolutional Network class.

# In[ ]:


class ConvNet(nn.Module):
    # Defining the layers of the convolutional network.
    def __init__(self,layers, c):
        super().__init__()

        # If we give a list of layers as input, we can also write it in the below way. 
        # self.layers = nn.ModuleList([ConvLayer(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.layer1 =  nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=2)
        self.layer2 =  nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=2)
        self.layer3 =  nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1)
        self.layer4 =  nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1)
        
#         self.out = nn.Linear(layers[-1], c)
        # c stands for the number of classes
        self.out = nn.Linear(80, c)
            
    # Defining the forward function.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))

        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.out(x)


# In[ ]:


# Number of neurons in the input layer.
n_inputs = 784
# Number of output layer neurons.
n_outputs = 10
# The Learning rate. We will also use learning rate decay in which after every 10 epochs, we decrease the learning rateby half
learning_rate = 0.001
# The number of epochs,
n_epochs = 30

model = ConvNet([1, 10, 20, 40, 80], 10).cuda()

# Defining the loss function
criterion = nn.CrossEntropyLoss()
# Defining the optimiser.
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-6)


# In[ ]:


# This function decreses the learning rate by half after every 10 epochs.
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = 0.001
    learning_rate = learning_rate * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
#     print("Epoch, ",epoch,  learning_rate)


# In[ ]:


if __name__ == "__main__":
    
    df_train = pd.read_csv('../input/train.csv')
    imageRotAngle = 20
    
    imageRotate = lambda mI: mI.rotate((2 * imageRotAngle * np.random.rand(1)) - imageRotAngle)
    # Creating augented data transformations
    train_aug_transformations = transforms.Compose([transforms.RandomRotation(10), transforms.Lambda(imageRotate), transforms.ToTensor()])
    test_transformations = transforms.Compose([transforms.ToTensor()])
    train_transformations = transforms.Compose([transforms.ToTensor()])
    
    custom_mnist_from_csv = CustomDatasetFromCSV_new('../input/train.csv', train_transformations)
    custom_mnist_aug_from_csv = CustomDatasetFromCSV_new('../input/train.csv', train_aug_transformations)
    custom_test_mnist_from_csv = CustomDatasetFromCSV_new('../input/test.csv', test_transformations, is_test=True)
    
    # Creating data loader with a batch size of 32 for training data + augmented training data.
    my_train_data_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset([custom_mnist_from_csv, custom_mnist_aug_from_csv]), batch_size=32, shuffle=False, num_workers=10)
    # Creating data loader with a batch size of 32 for testing data.
    #my_test_data_loader = torch.utils.data.DataLoader(dataset=custom_test_mnist_from_csv, batch_size=32, shuffle=False, num_workers=10)
    
    cnt_1 = 0
    for epoch in range(n_epochs):
        # adjusting learning rate in the beginning of the epoch
        adjust_learning_rate(optimizer, epoch)
        for i, (images, labels) in enumerate(my_train_data_loader):
            cnt_1 = cnt_1 + 1
            images = Variable(images.view(-1, 1, 28, 28)).cuda()
            labels = Variable(labels).cuda()
            
            output = model(images)
            train_loss = criterion(output, labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         predicted = predicted.cpu().numpy()
#         correct += (predicted == labels).sum()
#         print("Epoch : %d, Train Loss : %.4f, Accuracy = %0.3f" % (epoch, train_loss.abs(), correct/total))
        
    #
    ans_arr = []
    for i, images in enumerate(custom_test_mnist_from_csv):
        images = Variable(images.view(-1, 1, 28, 28)).cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        ans_arr.append(predicted)    
    print(len(ans_arr))
    print(cnt_1)


# # Creating the submission file

# In[ ]:



df_test = pd.read_csv("../input/test.csv")
ans_arr = [int(x) for x in ans_arr]
df_preds = pd.DataFrame()
df_preds['ImageId'] = pd.Series([i+1 for i in range(28000)])
df_preds['Label'] = pd.Series(ans_arr)
df_preds.to_csv("base.csv", index=False)

