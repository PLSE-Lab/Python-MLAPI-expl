#!/usr/bin/env python
# coding: utf-8

# **In this kernel, I try to implement a Chest X-Ray classifier by using a pretrained version of DenseNet. **

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to build Naive classifiero load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# I will use torch and torchvision to model my neural network and to take advantage of the ready-made optimization functions.

# In[ ]:


import torch
import torchvision
from torchvision import transforms
import torch.nn as nn


# This model was inspired by Rajpurkar et al., CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays
# with Deep Learning, 2017

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet121=torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier=nn.Sequential(
                nn.Linear(num_ftrs, 15), #don't forget to change the output
                nn.Sigmoid()
        )
    def forward(self, x):
        x=self.densenet121(x)
        return x
        


# All the labels of the data are placed in the csv file, that is where we have to read from.

# In[ ]:


df=pd.read_csv("../input/Data_Entry_2017.csv")
df.head()


# In the 'train_val_list.txt' file, the creator of the dataset specifies which of the files will be used for training and validation. So I obtain a list of the names of the scans which will be used for training and validation purposes. 

# In[ ]:


import random
with open('../input/train_val_list.txt') as f:
    content = f.readlines() #reading from the file
training_set = [x.strip() for x in content] #stripping away the newline characters
validation_set = random.sample(training_set, 17600) 
training_set = [i for i in training_set if i not in validation_set]
i1=training_set
print(len(training_set))
print(len(validation_set))


# Same reasoning to obtain the scans which will be part of the test set.

# In[ ]:


with open('../input/test_list.txt') as f:
    content = f.readlines()
test_set = [x.strip() for x in content]
print(len(test_set))


# In[ ]:


idxs=df['Image Index']
cnts=df['Finding Labels']
pathology_list = cnts.tolist()
pathology_list = set(pathology_list)
pathology_list = list(pathology_list)
cnts.index=idxs
print(cnts) #getting a view of what my dataset looks like


# This will determine the number of classes, which I have used to implement the linear layer of the neural network above.

# In[ ]:


pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','No Finding','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']
#Converting string classes to numbers, which can be used as labels for training
labels = df['Finding Labels'].tolist()
temp = []
for i, element in enumerate(labels):
    temp=len(pathology_list)*[0]
    for j, pathology in enumerate(pathology_list):
        if pathology in element:
            temp[j]=1
    labels[i] = temp
data_labels = pd.Series(labels, index=idxs)
data_labels.head()


# I obtain a pandas series with the labels of each set and the name of the picture files as the index.

# In[ ]:


training_labels = data_labels[training_set]
validation_labels = data_labels[validation_set]
test_labels = data_labels[test_set]
print(len(training_set))
print(len(validation_set))
print(len(test_set))


# Here I try to modify the index of the datasets in order for them to show the full path of the path of the picture files for each set.

# In[ ]:


import glob
import random
import cv2
import matplotlib.pyplot as plt

paths = []
nidxs1 = []
nidxs2=[]
nidxs3=[]

for filename in glob.iglob('../input/**/*.png', recursive=True): #worst algorithm ever
    paths.append(filename)
    for i in i1:
        if i in filename:
            nidxs1.append(filename)
    for i in validation_set:
        if i in filename:
            nidxs2.append(filename)
    for i in test_set:
        if i in filename:
            nidxs3.append(filename)
            
training_labels.index = nidxs1
validation_labels.index = nidxs2
test_labels.index=nidxs3

print(nidxs1[0])

s = random.sample(paths, 3)


plt.figure(figsize=(16,16))
plt.subplot(131)
plt.imshow(cv2.imread(s[0]))

plt.subplot(132)
plt.imshow(cv2.imread(s[1]))

plt.subplot(133)
plt.imshow(cv2.imread(s[2]));


# Now, I build my custom dataset and load the training data

# In[ ]:


#Custom dataset that will be used to load the data into a Dataloader
from PIL import Image
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, indices, labels, transforms=None):
        self.labels = labels
        self.indices = indices
        self.transforms = transforms
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        address = self.indices[index]
        x = Image.open(address).convert('RGB')
        y = torch.FloatTensor(self.labels[address])
        if self.transforms:
            x = self.transforms(x)
        return x, y


# In[ ]:


transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)


# In[ ]:


tl = training_labels.to_dict()
vl = validation_labels.to_dict()
trl = training_labels.to_dict()


# In[ ]:


dsetTrain = CustomDataset(nidxs1, tl, transform)
dsetVal = CustomDataset(nidxs2, vl, transform)
dsetTest = CustomDataset(nidxs3, trl, transform)


# Now the dataloaders are ready to be used in the training process.

# In[ ]:


trainloader = torch.utils.data.DataLoader(
    dataset = dsetTrain, batch_size = 10,
    shuffle = True, num_workers = 2
)

valloader = torch.utils.data.DataLoader(
    dataset = dsetVal, batch_size = 10,
    shuffle = True, num_workers = 2
)

testloader = torch.utils.data.DataLoader(
    dataset = dsetTest, batch_size = 10,
    shuffle = False, num_workers = 2
)


# In[ ]:


# The function that will train the model
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    i=0
    for epoch in range(num_epochs):
        for x, y in trainloader:
            x = torch.autograd.Variable(x).cuda()
            y = torch.autograd.Variable(y).cuda()
            scheduler.step()
            model.train()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            i+=1
            if(i==100):
                print('.')
                break #remove this line in real life
        print('Loss at the end of '+str(epoch+1)+': '+str(loss)+' '+str(i))
        break #remove this line in real life
        total = 0
        correct = 0
        for x, y in valloader:
            model.eval()
            x = torch.autograd.Variable(x).cuda()
            y = torch.autograd.Variable(y).cuda()
            with torch.no_grad():
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, y)
                total += y.size(0)
                correct += (pred==y).sum()
        epoch_acc = (correct/total)
        print('Accuracy at the end of '+str(epoch+1)+': '+str(epoch_acc*100)+'%')
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print('-'*5)
    print('Training finished...')
    print()
    model.load_state_dict(best_model_wts)
    return model


# In[ ]:


myNet = Net().cuda
criterion = nn.BCELoss().cuda()
optimizer_ft = torch.optim.Adam (myNet().parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
myNet = train_model(myNet(), criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)


# In[ ]:


#If you want to know the accuracy over the training set, uncomment the region below
#print('Checking accuracy over the test set')
#correct=0
#total=0
#with torch.no_grad():
#    for data in testloader:
#        x, y = data
#        scores=model_ft(x)
#        _, predicted=scores.data.max(1)
#        total+=y.size(0)
#        correct+=(predicted==y).sum()
#print('Accuracy over the test set is:  ', str((int(correct)/int(total))*100), ' %')


#  **Conclusion**
#  
#  Whenever the image had two or more labels, I used that as a separate class, which I think is not a good idea in the limited training set. I am very new to the field, so I am quite open to suggestions and corrections. Of course, the predicted loss is not real since we stop it after 100 iterations
