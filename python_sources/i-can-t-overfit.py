#!/usr/bin/env python
# coding: utf-8

# # I can't overfit...
# In this kernel I train very wide neural network without dropout and other regularization.  
# It still works ok and it barely overfits.

# In[ ]:


# Import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm_notebook as tqdm

get_ipython().run_line_magic('matplotlib', 'inline')

BATCH_SIZE = 256

print(os.listdir('../input/Kannada-MNIST/'))

# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# # Load data

# In[ ]:


df_train = pd.read_csv('../input/Kannada-MNIST/train.csv')
target = df_train['label']
df_train.drop('label', axis=1, inplace=True)

X_test = pd.read_csv('../input/Kannada-MNIST/test.csv')
X_test.drop('id', axis=1, inplace=True)

X_test.head()


# In[ ]:


X_train, X_dev, y_train, y_dev = train_test_split(df_train, target, stratify=target, random_state=42, test_size=0.2)
print('X_train', len(X_train))
print('X_dev', len(X_dev))
print('X_test', len(X_test))


# # Visualize

# In[ ]:


# Visualization Reference Kernel https://www.kaggle.com/josephvm/kannada-with-pytorch
# Some quick data visualization 
# First 10 images of each class in the training set

fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10,10))

# I know these for loops look weird, but this way num_i is only computed once for each class
for i in range(10): # Column by column
    num_i = X_train[y_train == i]
    ax[0][i].set_title(i)
    for j in range(10): # Row by row
        ax[j][i].axis('off')
        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')


# # PyTorch data generator

# In[ ]:


class CharData(Dataset):
    """
    Infer from standard PyTorch Dataset class
    Such datasets are often very useful
    """
    
    def __init__(self,
                 images,
                 labels=None,
                 transform=None,
                ):
        self.X = images
        self.y = labels
        
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = np.array(self.X.iloc[idx, :], dtype='uint8').reshape([28, 28, 1])
        if self.transform is not None:
            img = self.transform(img)
        
        if self.y is not None:
            y = np.zeros(10, dtype='float32')
            y[self.y.iloc[idx]] = 1
            return img, y
        else:
            return img


# In[ ]:


# Put some augmentation on training data
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor()
])

# Test data without augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# In[ ]:


# Create dataset objects
train_dataset = CharData(X_train, y_train, train_transform)
dev_dataset = CharData(X_dev, y_dev, test_transform)
test_dataset = CharData(X_test, transform=test_transform)


# In[ ]:


# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ## Show some generated samples

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=16, figsize=(30,4))

for batch in train_loader:
    for i in range(16):
        ax[i].set_title(batch[1][i].data.numpy().argmax())
        ax[i].imshow(batch[0][i, 0], cmap='gray')
    break


# # Define model

# In[ ]:


DEPTH_MULT = 4

class ConvLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.ops = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.ops(x)


class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        self.ops = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.ops(x)


def mixup(x, shuffle, lam, i, j):
    if shuffle is not None and lam is not None and i == j:
        x = lam * x + (1 - lam) * x[shuffle]
    return x


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.conv1 = ConvLayer(1, DEPTH_MULT * 32)
        self.conv2 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)
        self.conv3 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)
        self.conv4 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)
        
        self.conv5 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 64)
        self.conv6 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv7 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv8 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv9 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        self.conv10 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)
        
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = FCLayer(DEPTH_MULT * 64 * 7 * 7, DEPTH_MULT * 512)
        self.fc2 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)
        self.fc3 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)
        self.fc4 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)
        self.projection = nn.Linear(DEPTH_MULT * 512, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.mp(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.projection(x)
        
        return x


# In[ ]:


def criterion(input, target, size_average=True):
    """Categorical cross-entropy with logits input and one-hot target"""
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l


# In[ ]:


model = Net(10)
model = model.to(device)

n_epochs = 80

optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=n_epochs // 4, gamma=0.1)


# # Train model

# In[ ]:


def train(epoch, history=None):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
                
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                optimizer.state_dict()['param_groups'][0]['lr'],
                loss.data))
    exp_lr_scheduler.step()

def evaluate(epoch, history=None):
    model.eval()
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in dev_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss += criterion(output, target, size_average=False).data

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()
    
    loss /= len(dev_loader.dataset)
    accuracy = correct / len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
        history.loc[epoch, 'dev_accuracy'] = accuracy
    
    print('Dev loss: {:.4f}, Dev accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(dev_loader.dataset),
        100. * accuracy))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import gc\n\nhistory = pd.DataFrame()\n\nfor epoch in range(n_epochs):\n    torch.cuda.empty_cache()\n    gc.collect()\n    train(epoch, history)\n    evaluate(epoch, history)')


# # Plot losses

# In[ ]:


history['train_loss'].plot();


# In[ ]:


history.dropna()['dev_loss'].plot();


# In[ ]:


history.dropna()['dev_accuracy'].plot();


# In[ ]:


print('max', history.dropna()['dev_accuracy'].max())
print('max in last 5', history.dropna()['dev_accuracy'].iloc[-5:].max())
print('avg in last 5', history.dropna()['dev_accuracy'].iloc[-5:].mean())


# # Predict

# In[ ]:


model.eval()
predictions = []

for data in tqdm(test_loader):
    data = data.to(device)
    output = model(data).max(dim=1)[1] # argmax
    predictions += list(output.data.cpu().numpy())


# In[ ]:


submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission['label'] = predictions
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




