#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import tqdm
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import cv2
import albumentations
from albumentations import torch as AT


# In[ ]:


train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')


# In[ ]:


train_data['diagnosis'].value_counts().plot(kind='bar')


# In[ ]:


fig=plt.figure(figsize=(25,16))
for id in sorted(train_data['diagnosis'].unique()):
    for i,(idx,row) in enumerate(train_data.loc[train_data['diagnosis']==id].sample(6).iterrows()):
        ax=fig.add_subplot(6,6,id*6+i+1,xticks=[],yticks=[])
        im=Image.open(f"../input/train_images/{row['id_code']}.png")
        plt.imshow(im)
        ax.set_title(f'Label: {id}')


# In[ ]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder

def encode(y):
    le=LabelEncoder()
    integer_encoded=le.fit_transform(y)
    one_hot_encoder=OneHotEncoder(sparse=False)
    integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
    y=one_hot_encoder.fit_transform(integer_encoded)
    return y,le


# In[ ]:


y,le=encode(train_data['diagnosis'])


# In[ ]:


y.shape


# In[ ]:


class make_dataset(Dataset):
    def __init__(self,df,transform,y=None,datatype='train'):
        self.df=df
        self.datatype=datatype
        self.transform=transform
        self.image_file_list=[f'../input/{datatype}_images/{i}.png' for i in df['id_code'].values]
        if datatype=='train':
            self.labels=y
        else:
            self.labels = np.zeros((df.shape[0], 5))
        
    def __len__(self):
        return len(self.image_file_list)
    
    def __getitem__(self,idx):
        img_name=self.image_file_list[idx]
        img=Image.open(img_name)
        img=self.transform(img)
        label=self.labels[idx]
        
        if self.datatype=='train':
            return img,label
        else:
            return img,label,img_name
        
        


# In[ ]:


from skimage.transform import resize


# In[ ]:


tfms=transforms.Compose([transforms.Resize((224,224)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


# In[ ]:


dataset=make_dataset(df=train_data,transform=tfms,y=y)

tr, val = train_test_split(train_data.diagnosis, stratify=train_data.diagnosis, test_size=0.1)
train_sampler = SubsetRandomSampler(list(tr.index))
valid_sampler = SubsetRandomSampler(list(val.index))
batch_size = 64
num_workers = 0

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)


# In[ ]:


test_dataset=make_dataset(df=test_data,transform=transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),datatype='test')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


# In[ ]:


model_conv = torchvision.models.resnet50(pretrained=True)


# In[ ]:


num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(2048, 5)


# In[ ]:


model_conv.cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_conv.fc.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2, )


# In[ ]:


valid_loss_min = np.Inf
patience = 5
# current number of epochsimage = data_transforms(img)
where("validation", "loss", "didn't", "increase")
p = 0
# whether training should be stopped
stop = False

# number of epochs to train the model
import os
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

n_epochs = 20
for epoch in range(1, n_epochs+1):
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    train_auc = []

    for batch_i, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output, target.float())
        train_loss.append(loss.item())
        
        a = target.data.cpu().numpy()
        b = output[:,-1].detach().cpu().numpy()
        # train_auc.append(roc_auc_score(a, b))
        loss.backward()
        optimizer.step()
    
    model_conv.eval()
    val_loss = []
    val_auc = []
    for batch_i, (data, target) in enumerate(valid_loader):
        data, target = data.cuda(), target.cuda()
        output = model_conv(data)

        loss = criterion(output, target.float())

        val_loss.append(loss.item()) 
        a = target.data.cpu().numpy()
        b = output[:,-1].detach().cpu().numpy()
        # val_auc.append(roc_auc_score(a, b))

    # print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train auc: {np.mean(train_auc):.4f}, valid auc: {np.mean(val_auc):.4f}')
    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}.')
    
    valid_loss = np.mean(val_loss)
    scheduler.step(valid_loss)
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model_conv.state_dict(), 'model.pt')
        valid_loss_min = valid_loss
        p = 0

    # check if validation loss didn't improve
    if valid_loss > valid_loss_min:
        p += 1
        print(f'{p} epochs of increasing val loss')
        if p > patience:
            print('Stopping training')
            stop = True
            break        
            
    if stop:
        break


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')

model_conv.eval()
for (data, target, name) in test_loader:
    data = data.cuda()
    output = model_conv(data)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        sub.loc[sub['id_code'] == n.split('/')[-1].split('.')[0], 'diagnosis'] = le.inverse_transform([np.argmax(e)])
        
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head()

