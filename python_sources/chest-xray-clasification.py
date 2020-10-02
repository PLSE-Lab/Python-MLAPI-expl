#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("/kaggle/input"))


# In[ ]:


get_ipython().system('git clone https://github.com/sanchit2843/image_classification/')
get_ipython().run_line_magic('cd', 'image_classification')
#os.chdir('/kaggle/working/image_classification')


# In[ ]:


get_ipython().system('pip install torchsummary')
import dataloader,training,model
from predict import predict
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import models
from torch import nn
import numpy as np
import torch


# In[ ]:


im_size = 256
batch_size = 16
train_transforms = transforms.Compose([
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor()])
train_data = torchvision.datasets.ImageFolder(root = '/kaggle/input/chest_xray/chest_xray/train', transform = train_transforms)
train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)


# In[ ]:


#mean and standard deviation of custom data
mean,std = dataloader.normalization_parameter(train_loader)


# In[ ]:


#image transformations for train and test data
train_transforms = transforms.Compose([
                                        transforms.Resize((im_size,im_size)),
                                        transforms.RandomRotation(degrees=5),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
test_transforms = transforms.Compose([
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

#inverse normalization for image plot

inv_normalize =  transforms.Normalize(
    mean=-1*np.divide(mean,std),
    std=1/std
)


# In[ ]:


train_data = torchvision.datasets.ImageFolder(root = '/kaggle/input/chest_xray/chest_xray/train', transform = train_transforms)
test_data = torchvision.datasets.ImageFolder(root = '/kaggle/input/chest_xray/chest_xray/test', transform = test_transforms)
valid_data = torchvision.datasets.ImageFolder(root = '/kaggle/input/chest_xray/chest_xray/val', transform = test_transforms)

#label of classes

classes = train_data.classes
#encoder and decoder to convert classes into integer
decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]
#This will return 
dataloaders = dataloader.data_loader(train_data,encoder,test_data = test_data,valid_data = valid_data, batch_size = batch_size,inv_normalize = inv_normalize)


# In[ ]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        model = models.densenet121(pretrained = True)
        model = model.features
        for child in model.children():
          for layer in child.modules():
            layer.requires_grad = False
            if(isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d)):
              layer.requires_grad = True
        #model = EfficientNet.from_pretrained('efficientnet-b3')
        #model =  nn.Sequential(*list(model.children())[:-3])
        self.model = model
        self.linear = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, 2)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        out = self.model(x)
        avg_pool = nn.functional.adaptive_avg_pool2d(out, output_size = 1)
        max_pool = nn.functional.adaptive_max_pool2d(out, output_size = 1)
        out = torch.cat((avg_pool,max_pool),1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(self.bn1(out)))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return res


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier().to(device)


# In[ ]:


#Learning rate finder suggested in fast ai course
#model.lr_finder(classifier,dataloaders['train'])


# In[ ]:


lr = 0.001
training.train_model(classifier,dataloaders,encoder,inv_normalize,num_epochs=5,lr = lr,batch_size = batch_size,patience = None,classes = classes)


# In[ ]:


for param in classifier.parameters():
    param.requires_grad = True


# In[ ]:


lr = 0.0005
training.train_model(classifier,dataloaders,encoder,inv_normalize,num_epochs=1,lr = lr,batch_size = batch_size,patience = 3,classes = classes)

