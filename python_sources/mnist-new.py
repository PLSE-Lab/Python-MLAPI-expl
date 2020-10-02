#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset
import numpy as np
from time import time
import pandas as pd
import collections

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


batch_size = 16

class trainSet(Dataset):
  def __init__(this, imdat, lbdat, transform=None):
    
    this.imgs = imdat
    this.lbls = lbdat
    this.transform = transform
  
  def __len__(this):
    return(len(this.lbls))
    
  def __getitem__(this, index):
    return this.imgs[index], this.lbls[index]
  
iiis = train.drop(labels = ["label"],axis = 1)
llls = train["label"].values.tolist()

iiis = iiis / 255.0
iiis = iiis.values.tolist()

tr_Setup = trainSet(iiis, llls)
tr_data = DL(tr_Setup, batch_size=batch_size, shuffle=True, drop_last=True)


# In[ ]:


class ConvNet(nn.Module):
    def __init__(this, HL):
        super(ConvNet, this).__init__()
        
        this.Layer1 = nn.Sequential()
        this.Layer1.add_module("Convolution Layer 1", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1))
        this.Layer1.add_module("Activation Layer 1" , nn.ReLU())
        this.Layer1.add_module("Convolution Layer 2", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        this.Layer1.add_module("Activation Layer 2" , nn.ReLU())
        this.Layer1.add_module("Pooling Layer"      , nn.MaxPool2d(kernel_size=2))

        this.Layer2 = nn.Sequential()
        this.Layer2.add_module("Convolution Layer 1", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        this.Layer2.add_module("Activation Layer 1" , nn.ReLU())
        this.Layer2.add_module("Convolution Layer 2", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        this.Layer2.add_module("Activation Layer 2" , nn.ReLU())
        this.Layer2.add_module("Pooling Layer"      , nn.MaxPool2d(kernel_size=2))

        this.Layer3 = nn.Sequential()
        this.Layer3.add_module("Convolution Layer 1", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        this.Layer3.add_module("Activation Layer 1" , nn.ReLU())
        this.Layer3.add_module("Convolution Layer 2", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        this.Layer3.add_module("Activation Layer 2" , nn.ReLU())
        this.Layer3.add_module("Pooling Layer"      , nn.MaxPool2d(kernel_size=2))

        this.LC = nn.Sequential()
        this.LC.add_module("Fully Connected Layer 1", nn.Linear(256*3*3, HL))
        this.LC.add_module("Activation Layer FC1", nn.ReLU())
        this.LC.add_module("Fully Connected Layer 2", nn.Linear(HL, HL))
        this.LC.add_module("Activation Layer FC2", nn.ReLU())
        this.LC.add_module("Fully Connected Layer 2", nn.Linear(HL, 10))
        this.LC.add_module("Activation Layer FC3", nn.LogSoftmax(dim=1))

    def classify(this, x):
        x = this.Layer1(x)
        x = this.Layer2(x)
        x = this.Layer3(x)
        x = x.view(x.shape[0], -1)
        x = this.LC(x)
        return x
    
HiL = 4096
Network = ConvNet(HiL)
Network = Network.cuda()

lossFunc = nn.NLLLoss()


# In[ ]:


LearnRate, Momentum  = 0.001, 0.9
lossList = []

optimizer = optim.SGD(Network.parameters(), LearnRate, Momentum)

no_of_passes = 50
begin_time = time()
for p in range(no_of_passes):
    start_time = time()
    lossPerPass = 0
    for images, labels in tr_data:
        images = torch.stack(images)
        images = torch.t(images)
        images = images.view(images.shape[0], 1, 28, 28)
        images = images.float()
        images = images.cuda()
        
        labels = labels.cuda()

        optimizer.zero_grad()
        output = Network.classify(images)
        loss = lossFunc(output, labels)
        lossPerPass += loss.item()
        loss.backward()
        optimizer.step()
    else:
        print("\nPass          : ", p+1)
        print("Training loss : " , lossPerPass/len(tr_data))
        print("Time taken    : ", time() - start_time, " seconds\n")
        lossList.append(lossPerPass/len(tr_data))

print("\nTotal Training Time : ", time() - begin_time, "seconds\n")


# In[ ]:


batch_size = 1

class testSet(Dataset):
  def __init__(this, imdat, transform=None):
    
    this.imgs = imdat
    this.transform = transform
  
  def __len__(this):
    return(len(this.imgs))
    
  def __getitem__(this, index):
    return this.imgs[index]

test = test / 255.0
iiis = test.values.tolist()

ts_Setup = testSet(iiis)
ts_data = DL(ts_Setup, batch_size=batch_size, shuffle=False, drop_last=False)


# In[ ]:


predList = []

for images in ts_data:
    images = torch.stack(images)
    images = torch.t(images)
    images = images.view(images.shape[0], 1, 28, 28)
    images = images.float()
    images = images.cuda()

    with torch.no_grad():
        logProb = Network.classify(images)
    
    Prob = torch.exp(logProb).cpu()
    Prob = np.array(Prob.numpy()[0:batch_size])

    for i in range(batch_size):
        P = list(Prob[i])
        prediction = P.index(max(P))
        predList.append(prediction)


# In[ ]:


predList = pd.Series(predList,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predList],axis = 1)
submission.to_csv("mnistPredictions.csv",index=False)


# In[ ]:


DSPath = "./data"

MNIST_mean = 0.1307
MNIST_std  = 0.3081

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_mean,), (MNIST_std,)),])
ts_data_fromTorch = DL(datasets.MNIST(DSPath, download=True, train=False, transform=transforms.ToTensor()), batch_size=64, shuffle=True)

correct = 0
total = 0
for images, labels in ts_data_fromTorch:
    images = images.cuda()
    labels = labels.cuda()
    with torch.no_grad():
        logProb = Network.classify(images)
    
    Prob = torch.exp(logProb).cpu()
    Prob = np.array(Prob.numpy()[0:len(labels)])
    labels = labels.cpu()
    for i in range(len(labels)):
        P = list(Prob[i])
        prediction = P.index(max(P))
        groundtruth = labels.numpy()[i]
        if groundtruth == prediction:
            correct += 1
        total += 1    

print("\n\nModel Test Accuracy : ", (correct/total) * 100, "%")

