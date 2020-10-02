#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


# In[ ]:


PATH = Path('../input/')


# In[ ]:


def return_label(fileName):
    df = pd.read_csv(filepath_or_buffer=PATH/fileName)
    if fileName!='test.csv':
        label = np.array(df['label'])
        data = np.array(df[df.columns[1:]], dtype=np.float)
        new_data = np.reshape(a=data, newshape=(data.shape[0], 28, 28))
        return new_data, label
    else:
        data = np.array(df, dtype=np.float)
        new_data = np.reshape(a=data, newshape=(data.shape[0], 28, 28))
        return new_data


# In[ ]:


trainData, trainLabel = return_label('train.csv')
testData = return_label('test.csv')


# In[ ]:


trainData = trainData/255.
trainData = (trainData-0.5)/0.5

testData = testData/255.
testData = (testData-0.5)/0.5

trainData = torch.from_numpy(trainData).cuda()
testData = torch.from_numpy(testData).cuda()
trainData, testData = trainData.type(torch.FloatTensor), testData.type(torch.FloatTensor)


# In[ ]:


trainData


# In[ ]:


trainData.shape


# In[ ]:


trainData = trainData.unsqueeze_(dim=1)
testData = testData.unsqueeze_(dim=1)


# In[ ]:


trainData.shape


# In[ ]:


trainDataset = torch.utils.data.TensorDataset(trainData, torch.from_numpy(trainLabel))
train_dl = torch.utils.data.DataLoader(trainDataset, batch_size=100, shuffle=False, num_workers=4)

test_dl = torch.utils.data.DataLoader(testData, batch_size=100, shuffle=False, num_workers=4)


# In[ ]:


print(len(trainData), len(testData))


# In[ ]:


def num_classes(loader):
    classCount = [0]*10
    for batch_id, (images, labels) in enumerate(loader):
        for label in labels:
            classCount[int(label)]+=1
    return classCount


# In[ ]:


classes = [i for i in range(10)]
classCount = num_classes(train_dl)
fig, ax = plt.subplots()
ax.barh(y=classes, width=classCount)
ax.set_xlabel('# of Examples')
ax.set_ylabel('# of Digit Class')
ax.set_title('Train Set')


# In[ ]:


temp = train_dl.dataset[0][0].numpy()
temp = np.reshape(temp, (temp.shape[1], temp.shape[2]))
plt.imshow(temp)


# In[ ]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 1)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


n1 = Model().cuda()
n1


# In[ ]:


def train(model, mode, decay, criterion, dataloader, optimizer, dictionary, num_epochs=30):    
    totalLoss = []
    totalLRs = []
    correct = 0
    total = 0
    LR = 0
    for epoch in range(num_epochs):
        if(decay == True):
            for param in optimizer.param_groups:
                LR = param['lr'] * (0.1**(epoch//7))
                param['lr'] = LR
            totalLRs.append(LR)
        print("Epoch = {}/{} ".format(epoch,num_epochs),end=" ")
        for batch_id,(image, label) in enumerate(dataloader):
            if(mode == True):
                optimizer.zero_grad()
                image = torch.autograd.Variable(image)
                label = torch.autograd.Variable(label)
                image = image.cuda()
                label = label.cuda()
            else:
                image = torch.autograd.Variable(image)
                image = image.cuda()
            output = model.forward(image)
            if(mode == True):
                loss = criterion(output,label)
            _, predictated = torch.max(output.data,1)   
            if(mode == True):
                correct += (predictated == label.data).sum()
                total += label.size(0)
                loss.backward()
                optimizer.step()
            del image,label
        torch.cuda.empty_cache()
        print("Loss = {:.5f}".format(loss.item()))
        totalLoss.append(loss.item())
    dictionary['totalLoss'] = totalLoss
    dictionary['correct'] = correct
    dictionary['totalSize'] = total
    dictionary['totalLRs'] = totalLRs
    return model,dictionary


# In[ ]:


optimizer = torch.optim.SGD(n1.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss().cuda()


# In[ ]:


dictModel = {}
n1, dictModel = train(model=n1, mode=True, decay=True, 
                      criterion=criterion, dataloader=train_dl, optimizer=optimizer, 
                      dictionary=dictModel, num_epochs=50)


# In[ ]:


plt.plot(dictModel['totalLoss'])


# In[ ]:


optimizer = torch.optim.SGD(n1.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss().cuda()

dictModel = {}
n1, dictModel = train(model=n1,mode=True,decay=False,criterion=criterion,dataloader=train_dl,
                            optimizer=optimizer,dictionary=dictModel,num_epochs=20)


# In[ ]:


plt.plot(dictModel['totalLoss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Dataset')


# In[ ]:


avgLossTest = []
totalPrediction = []
for id, image in enumerate(test_dl):
    image = torch.autograd.Variable(image).cuda()
    output = n1(image)
    _, predictated = torch.max(output.data, 1)
    totalPrediction.append(predictated)


# In[ ]:


len(totalPrediction)


# In[ ]:


len(testData)


# In[ ]:


plt.imshow(test_dl.dataset[0][0]), totalPrediction[0][0]


# In[ ]:


test_dl.dataset


# In[ ]:


plt.imshow(test_dl.dataset[1][0]), totalPrediction[1][0]


# In[ ]:


temp = [list(x.cpu().numpy()) for x in totalPrediction]
Label = []
for x in temp:
    for y in x:
        Label.append(y)
ImageId = [t for t in range(1, 28001)]
len(ImageId)
dfDict = {'ImageId':ImageId, 'Label':Label}
df = pd.DataFrame(dfDict)
df.to_csv('submission.csv', index=False)


# In[ ]:




