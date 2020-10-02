#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# In[ ]:


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root='./mnist', train=True, transform=trans, download=True)
test_set = dset.MNIST(root='./mnist', train=False, transform=trans)

batch_size = 128

#divide the set to training and validation
from torch.utils.data.sampler import SubsetRandomSampler
num_train = len(train_set)
valid_size = 0.1
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,                 
                 sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,                 
                 sampler=valid_sampler)


test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1,
                shuffle=False)

print('number of training data:', len(train_set)-split)
print('number of validation data:', split)
print('number of test data:', len(test_set))


# In[ ]:


class LeNet(nn.Module):
    def __init__(self,n_class=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 20,
            kernel_size = 3
        )
        self.conv2 = nn.Conv2d(
            in_channels = 20,
            out_channels = 50,
            kernel_size = 3
        ) 
        self.fc1 = nn.Linear(11*11*50, n_class)        
    def forward(self, x):
        x = F.relu(self.conv1(x))   # x:[batch_size,1,28,28] => x:[batch_size,20, 24, 24]
        x = F.max_pool2d(x, 2, 2)   # x:[batch_size,20,24,24] => x:[batch_size,20, 12, 12]
        x = F.relu(self.conv2(x))   # x:[batch_size,20,12,12] => x:[batch_size,50, 8, 8]        
        x = x.view(-1, 11*11*50)      # x:[batch_size,50,4,4] => x:[batch_size,50*4*4]        
        x = self.fc1(x)             # x:[batch_size,50*4*4] => x:[batch_size,10]
        return x

model = LeNet()
print(model)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5

train_loss = []
for epoch in xrange(num_epochs):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x.requires_grad_()
        logits = model(x)        
        loss = criterion(logits, target)
        ave_loss = ave_loss * 0.9 + loss * 0.1
        train_loss.append(loss)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss)
    
    # validation set
    preds = []
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(valid_loader):        
        logits = model(x)
        loss = criterion(logits, target)
        _, pred_label = torch.max(logits, 1)
        preds.append(pred_label)
        total_cnt += x.size()[0]
        correct_cnt += (pred_label == target).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(valid_loader):
            print '==>>> epoch: {}, batch index: {}, validation loss: {:.6f}, validation acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt.item() * 1.0 / total_cnt)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.arange(len(train_loss)), train_loss)
plt.show()


# In[ ]:


# testing
preds = []
correct_cnt = 0
total_cnt = 0.0
for batch_idx, (x, target) in enumerate(test_loader):        
    logits = model(x)
    _, pred_label = torch.max(logits, 1)
    preds.append(pred_label)
    total_cnt += x.size()[0]
    correct_cnt += (pred_label == target).sum()

    if(batch_idx+1) % 1000 == 0 or (batch_idx+1) == len(test_loader):
        print '==>>> #test_samples: {}, acc: {:.3f}'.format(batch_idx+1, correct_cnt.item() * 1.0 / total_cnt)


# In[ ]:


#submission: write to the file
with open('submission.csv','wb') as file:
    file.write('Id,Label\n')
    for idx, lbl in enumerate(preds): 
        line = '{},{}'.format(idx,lbl.item())
        file.write(line)
        file.write('\n')


# In[ ]:


# save model
torch.save(model.state_dict(), 'best_model.t7')

