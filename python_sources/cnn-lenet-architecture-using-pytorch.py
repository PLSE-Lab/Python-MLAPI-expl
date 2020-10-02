#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


import torchvision
import torchvision.transforms as transforms


# In[ ]:


class Dataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


# In[ ]:


class DatasetMNIST2(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


# In[ ]:


trainset = DatasetMNIST2('../input/train.csv', transform=torchvision.transforms.ToTensor())


# In[ ]:


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


# In[ ]:


dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)

print(images[1].shape)
print(labels[1].item())


# In[ ]:


img = images[1]
print(type(img))


# In[ ]:


npimg = img.numpy()
print(npimg.shape)


# In[ ]:


npimg = np.transpose(npimg, (1, 2, 0))
print(npimg.shape)


# In[ ]:


import torch.nn as nn
import torch.optim as optim


# In[ ]:


class LeNet(nn.Module):
    def __init__(self): 
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, 5),         # (N, 1, 28, 28) -> (N,  6, 24, 24)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 24, 24) -> (N,  6, 12, 12)
            nn.Conv2d(6, 16, 5),        # (N, 6, 12, 12) -> (N, 16, 8, 8)  
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)   # (N,16, 8, 8) -> (N, 16, 4, 4)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(256,120),         # (N, 256) -> (N, 120)
            nn.ReLU(),
            nn.Linear(120,84),          # (N, 120) -> (N, 84)
            nn.ReLU(),
            nn.Linear(84,10)            # (N, 84)  -> (N, 10)
        )
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x
    
    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred


# In[ ]:


net = LeNet()
out = net(images)


# In[ ]:


print(out)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


def evaluation(dataloader):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total


# In[ ]:


net = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters())


# In[ ]:


batch_size = 128
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


# In[ ]:


max_epochs = 16

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        
    print('Epoch: %d/%d, Train acc: %0.2f' % (epoch, max_epochs, evaluation(trainloader)))
        


# In[ ]:


testset = np.loadtxt('../input/test.csv',skiprows=1,delimiter=",")


# In[ ]:


testset = testset.reshape(testset.shape[0],28,28,1)


# In[ ]:


testset = testset/255


# In[ ]:


testset.shape


# In[ ]:


testset = np.transpose(testset, (0, 3, 1,2))


# In[ ]:


testset.shape


# In[ ]:


testset = torch.tensor(testset)


# In[ ]:


testset = testset.float()


# In[ ]:


testset = testset.to(device)


# In[ ]:


Y_pred_val = net.predict(testset)


# In[ ]:


pred = torch.argmax(Y_pred_val, dim=1)


# In[ ]:


pred.shape


# In[ ]:


pred = pred.to('cpu')


# In[ ]:


results = pd.Series(pred , name = 'Label')


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = 'ImageId'),results],axis=1)


# In[ ]:


submission.to_csv("final_submission.csv",index=False)

