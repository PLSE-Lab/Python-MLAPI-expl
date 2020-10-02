#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


# In[ ]:


train_image = pd.read_csv("../input/ahcd1/csvTrainImages 13440x1024.csv", header = None)
train_label = pd.read_csv("../input/ahcd1/csvTrainLabel 13440x1.csv", header = None)
test_image = pd.read_csv("../input/ahcd1/csvTestImages 3360x1024.csv", header = None )
test_label = pd.read_csv("../input/ahcd1/csvTestLabel 3360x1.csv", header = None )


# In[ ]:


# add the lables of the dataset
train_image['label'] = train_label
test_image['label'] = test_label

print(train_image.shape)


# In[ ]:


class ImageDataset(Dataset):
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, self.data.columns != 'label'].values.astype(np.uint8).reshape(32, 32)
        label = self.data.iloc[index, -1] - 1
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


# In[ ]:


# uint8 	 Unsigned integer (0 to 255)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_image_dataset = ImageDataset(train_image, transform)


# In[ ]:


image , label = train_image_dataset.__getitem__(32)

image.size()


# In[ ]:


loader = DataLoader(train_image_dataset, batch_size= 32, shuffle=True)


# In[ ]:


class ConvolutionNNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1024, 1000)
        self.fc2 = nn.Linear(1000, 28)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
# Train the model
num_epochs = 32
learning_rate = 0.001
total_step = len(loader)
loss_list = []
acc_list = []
model = ConvolutionNNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate);
criterion = nn.CrossEntropyLoss();

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


# In[ ]:


test_image_dataset = ImageDataset(test_image, transform)
test_loader = DataLoader(test_image_dataset, batch_size= 1, shuffle=False)
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for (images, labels) in test_loader:
#         print(images.shape, labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 3360 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
# torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

