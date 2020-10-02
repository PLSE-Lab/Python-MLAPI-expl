#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#we need to import some libraries
import numpy as np
import pandas as pd
import os
import sys
print(os.listdir("../input"))
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import torch.backends.cudnn as cudnn
from time import time
from tqdm import tqdm


# In[ ]:


# load image data
dataFile = pd.read_csv('../input/train.csv')
dataFile.head()


# In[ ]:


class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', dataFile=None, transform=None, labelArray=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.labelArray = labelArray
        if self.datatype == 'train':
            self.dataFile = dataFile.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.dataFile[idx][0])
            label = self.labelArray[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((5005,))
        
        img = Image.open(img_name).convert('RGB')
        image = self.transform(img)
        
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]


# In[ ]:


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded.shape)

    y = onehot_encoded
    #print(y.shape)
    return y, label_encoder

y, label_encoder = prepare_labels(dataFile['Id'])


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet101(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5005)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        


# In[ ]:


input_size = 224 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
data_transforms =  transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
train_dataset= WhaleDataset(datafolder='../input/train/', datatype='train', 
                            dataFile=dataFile, transform=data_transforms, 
                            labelArray=y)
dset_loaders = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=0, pin_memory=True)
N_train = len(y)
print(N_train)


# In[ ]:


batch_size = 32
num_epochs = 5
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    
    running_loss, running_corrects, tot = 0.0, 0.0, 0.0
    ########################
    model.train()
    torch.set_grad_enabled(True)
    ## Training 
    for batch_idx, (inputs, labels) in enumerate(dset_loaders):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels.float())
        running_loss += loss*inputs.shape[0]
        loss.backward()
        optimizer.step()
        ############################################
        _, preds = torch.max(outputs.data, 1)
        _, tmplabel = torch.max(labels.data, 1)

        running_loss += loss.item()
        running_corrects += preds.eq(tmplabel).cpu().sum()
        tot += labels.size(0)
        sys.stdout.write('\r')
        try:
            batch_loss = loss.item()
        except NameError:
            batch_loss = 0

        top1error = 1 - float(running_corrects)/tot
        if batch_idx % 100 == 0:
            sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\n'
                             % (epoch + 1, num_epochs, batch_idx + 1,
                            (len(os.listdir('../input/train')) // batch_size), batch_loss/batch_size))
            sys.stdout.flush()
            sys.stdout.write('\r')
        
    #accuracy = float(running_corrects)/N_train
    epoch_loss = running_loss/N_train

    print('\n| Training loss %.4f'            % (epoch_loss))


# In[ ]:


#Evaluate and predict test data
sub = pd.read_csv('../input/sample_submission.csv')
test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

test_set = WhaleDataset(
    datafolder='../input/test/', 
    datatype='test', 
    transform=test_transforms
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=0, pin_memory=True)

model.eval()
for (inputs, labels, name) in tqdm(test_loader):
    inputs = inputs.to(device)
    output = model(inputs)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(label_encoder.inverse_transform(e.argsort()[-5:][::-1]))
print(output.shape)
sub.to_csv('submission.csv', index=False)

