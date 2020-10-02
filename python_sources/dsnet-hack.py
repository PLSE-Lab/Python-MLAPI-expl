#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import os
import cv2
import math
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm_notebook

#deeplearning
import time
import torch
import torchvision
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0")


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('ls /kaggle/input/dsnet-kaggledays-hackathon/train/train')


# In[ ]:


BASE_PATH = "../input/dsnet-kaggledays-hackathon/train/train"
df_arr = []
for dirname in os.listdir(BASE_PATH):
    for filename in os.listdir(os.path.join(BASE_PATH, dirname)):
        df_dict = {"y": dirname, "path": os.path.join(BASE_PATH,dirname,filename)}
        df_arr.append(df_dict)


# In[ ]:


df = pd.DataFrame(df_arr)


# In[ ]:


len(df)


# In[ ]:


img = plt.imread(df['path'][0])
plt.axis('off')
plt.imshow(img)
plt.show()


# In[ ]:


plt.figure(figsize=(20,2))
df['y'].value_counts().plot(kind='bar')


# In[ ]:


dicty =  df['y'].value_counts().sort_values()[0:25].index.values


# In[ ]:


dicty = {i:100 for i in dicty }


# In[ ]:


dicty


# In[ ]:


from imblearn.over_sampling import RandomOverSampler


sm = RandomOverSampler(sampling_strategy=dicty, random_state=42)

# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_sample(df['path'].values.reshape(-1,1), df['y'])
oversampled_train = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
oversampled_train.columns = df.columns


# In[ ]:


plt.figure(figsize=(20,2))
oversampled_train['y'].value_counts().plot(kind='bar')


# In[ ]:


df = oversampled_train


# In[ ]:


def preprocess_image(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (380, 380))
    return image


# In[ ]:


len(df)


# In[ ]:


N =  len(df)
x_train = np.empty((N, 380,380, 3), dtype=np.uint8)
for i, path in enumerate(tqdm(df['path'])):
    x_train[i, :, :, :]= preprocess_image(path)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()


# In[ ]:


y_train = number.fit_transform(df['y'])
cls = number.classes_


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.14,random_state = 42)


# In[ ]:


transform = {'train' : transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(360),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
}


# In[ ]:


class ARTWORK(Dataset):
    
    def __init__(self,x_train,y_train,transform=None,train=True):
        self.x = x_train
        self.y = y_train
        self.transform = transform
        self.train=train
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.train==True:
            image = self.x[idx].astype(np.uint8).reshape((380,380,3))
            label = self.y[idx]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return {'image': image,
                'label': label
                }


# In[ ]:


data = ARTWORK(x_train,y_train,transform=transform['train'],train=True)


# In[ ]:


valid = ARTWORK(x_val,y_val,transform=transform['valid'],train=True)


# In[ ]:


x = data.__getitem__(12)
image = x['image'].numpy()
image = np.transpose(image,[2,1,0])
plt.imshow(image)
plt.title(cls[x['label']])
plt.show()


# In[ ]:


get_ipython().system('pip install efficientnet-pytorch')


# In[ ]:


from efficientnet_pytorch import EfficientNet


# In[ ]:


get_ipython().system('ls ../input/efficientnet-pytorch')


# In[ ]:


import torchvision.models as models


# In[ ]:


model = EfficientNet.from_name('efficientnet-b4')
model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth'))
model._fc = nn.Sequential(
     nn.Linear(in_features=1792, out_features=49, bias=True),
     nn.LogSoftmax()
    )

model = model.to(device)


# In[ ]:


data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True, num_workers=4)
valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=16, shuffle=False, num_workers=4)


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)


# In[ ]:


since = time.time()
criterion = nn.NLLLoss()
num_epochs = 25
previous_loss = 100
lossy = []
accy = []
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    scheduler.step()
    model.train()
    running_loss = 0.0
    tk0 = tqdm(data_loader, total=int(len(data_loader)))
    counter = 0
    train_acc  = []
    for bi, d in enumerate(tk0):
        inputs = d["image"]
        labels = d["label"]
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = torch.max(outputs,1)[1]
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            acc = accuracy_score(outputs,labels)
            train_acc.append(acc)
        running_loss += loss.item()
        counter += 1
    epoch_loss = running_loss / len(data_loader)
    print('train acc {:.4f}'.format(np.mean(train_acc)) )
    print('Training Loss: {:.4f}'.format(epoch_loss))
    model.eval()
    val_loss = []
    correct = []

    with torch.no_grad():
        for bi,d in enumerate(tqdm(valid_data_loader,total=int(len(valid_data_loader)))):
            inputs = d["image"]
            labels = d["label"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            outputs = torch.max(outputs,1)[1]
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            acc = accuracy_score(outputs,labels)
           
            correct.append(acc)
            val_loss.append(loss.item())
            
            
    
    print('Validation Loss: {:.4f}'.format(np.mean(val_loss)))
    print('Accuracy score: {:.4f}'.format(np.mean(correct)))
    if np.mean(val_loss)<previous_loss:
        previous_loss = np.mean(val_loss)
        print('LOSS improved')
    lossy.append(np.mean(val_loss))
    accy.append(np.mean(correct))


# In[ ]:


torch.save(model.state_dict(), "model_dsnet_1.pth")


# <a href="./model_dsnet_1.pth"> Download File </a>

# In[ ]:


sub = pd.read_csv("../input/dsnet-kaggledays-hackathon/sample_submission.csv")


# In[ ]:


sub.head()


# In[ ]:


len(sub)


# In[ ]:


N =  len(sub)
x_test= np.empty((N, 380, 380, 3), dtype=np.uint8)
for i, path in enumerate(tqdm(sub['id'])):
    x_test[i, :, :, :]= preprocess_image(f'../input/dsnet-kaggledays-hackathon/test/test/{path}')


# In[ ]:


class TEST(Dataset):
    
    def __init__(self,x_train,transform=None,train=True):
        self.x = x_train
        self.transform = transform
        self.train=train
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.train==True:
            image = self.x[idx].astype(np.uint8).reshape((380, 380,3))
        
        if self.transform is not None:
            image = self.transform(image)
            
        return {'image': image
                }


# In[ ]:


test = TEST(x_test,transform=transform['valid'],train=True)


# In[ ]:


data_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=4)


# In[ ]:


model.eval()
pred = []
with torch.no_grad():
    for i, data in enumerate(tqdm(data_loader,total=int(len(data_loader)))):
        images = data['image']
        images = images.to(device, dtype=torch.float)
        predict = model(images)
        predict = torch.max(predict,1)[1].cpu().squeeze().numpy().reshape(-1)[0]
        pred.append(predict)
        


# In[ ]:


torch.max(model(images),1)[1].detach().cpu().squeeze().numpy().reshape(-1)[0]


# In[ ]:


number.classes_[pred]


# In[ ]:



sub['predicted_class'] = number.classes_[pred]
sub.to_csv("submission.csv",index=False)


# In[ ]:


sub['predicted_class'].value_counts()


# <a href="./submission.csv"> Download File </a>

# In[ ]:




