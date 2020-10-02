#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import warnings
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset, DataLoader,Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import copy 
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Any results you write to the current directory are saved as output.
use_cuda = torch.cuda.is_available()
sns.set(style="darkgrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not use_cuda:
    print('No GPU found. Please use a GPU to train your neural network.')


# In[ ]:


from PIL import Image
# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
imageFile = "/kaggle/input/mydatasetforaerialcactus/preprocess_v1/preprocess/train/Large/clean1922.png"
im1 = Image.open(imageFile)
# adjust width and height to your needs
width = 224
height = 224
# use one of these filter options to resize the image
im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
im5 = im1.resize((width, height), Image.ANTIALIAS) 

display(im5)


# In[ ]:


def convertDate(x,y):
    print(x)
    if x != "":
        return path/x
    return x


# In[ ]:


path = Path('/kaggle/input/mydatasetforaerialcactus/preprocess_v1/preprocess')

df_train = pd.read_csv(path/'train.csv', low_memory=False)
df_test = pd.read_csv(path/'test.csv', low_memory=False)


# In[ ]:


#df_train = df_train[:18528]
#_File"] = df_train[["Image_File", "Class"]].applymap(convertDate)
#.apply(lambda se: se.str.zfill(4))
print(df_train.shape)
print(df_test.shape)
df_train.head()


# In[ ]:


plt.figure(figsize=(6, 26))
sns.countplot(x="Class", data=df_train)


# In[ ]:


# NOTE: class is inherited from Dataset
class LumarImageLabelDataset(Dataset):
    def __init__(self, df_data, prediction, folder="train", transformtype="train"):
        super().__init__()
        self.df = df_data
        self.labels_dict = {"Large":0, "Small":1}
        self.prediction = prediction
        self.folder = folder
        self.transformtype = transformtype

    def __len__(self):
        return len(self.df)
    
    @property
    def train_labels(self):
        #warnings.warn("train_labels has been renamed targets")
        return self.prediction
    
    def __getitem__(self, index):
        #print(index)
        label = self.prediction[index]
        #print(label)
        tensorimage = self.preprocess_image(self.df[index], label)
        #label_tensor = self.get_dummies(label)
        #print(tensorimage)
        label_tensor = int(self.labels_dict[label])
        return tensorimage, label_tensor    
   
    def preprocess_image(self, img_path, label):
        data_transform = transforms.Compose([transforms.ToPILImage(),
                                             #transforms.Resize(256),
                                             #transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomRotation(degrees=40), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        
        test_transform = transforms.Compose([transforms.ToPILImage(),
                                             #transforms.Resize(256),
                                             #transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        image = None
        path = Path('/kaggle/input/mydatasetforaerialcactus/preprocess_v1/preprocess')
        if self.transformtype == 'train':
            #print(path/"{}/{}/{}".format(self.folder, label, img_path))
            image = Image.open(path/"{}/{}/{}".format(self.folder, label, img_path)).convert('RGB') 
            #image = cv2.imread(path/"{}/{}/{}".format(self.folder, label, img_path), 0)
            image = data_transform(np.array(image))
        else:
            image = Image.open(path/"{}/{}".format(self.folder, img_path)).convert('RGB') 
            #image = cv2.imread(path/"{}/{}".format(self.folder, img_path))
            image = data_transform(np.array(image))
            
        return image
    
    def get_dummies(self, attribute_id):
        label_tensor = torch.zeros((1, 102))
        #label_tensor = torch.zeros((1, 1))
        #for label in attribute_id.split():
        label_tensor[0, int(attribute_id)-1] = 1
        #label_tensor[0, 0] = attribute_id
        return label_tensor


# In[ ]:


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))             if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)             if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is ImageLabelDataset:
            return dataset.train_labels[idx].item()
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))


# In[ ]:


# df_train.head()
batch_size = 64
train_image = df_train["Image_File"]
#target = df_train.drop(['id', 'attribute_ids'],axis=1)
target = df_train["Class"]

df_test["Class"] = "Large"

X_train, X_val, y_train, y_val = train_test_split(train_image, target, test_size=0.33)

#smt = SMOTE()
#X_train, y_train = smt.fit_sample(X_train.reshape(-1, 1), y_train)
#print(np.bincount(y_train))

test_image = df_test["Image_File"]
test_target = df_test["Class"]
#test_target = df_test.drop(['id', 'attribute_ids'],axis=1)

train_set = LumarImageLabelDataset(df_data=X_train.values, prediction=y_train.values, 
                              folder="train", transformtype="train")
val_set = LumarImageLabelDataset(df_data=X_val.values, prediction=y_val.values, 
                            folder="train", transformtype="train")
predict_set = LumarImageLabelDataset(df_data=test_image, prediction=test_target, 
                                folder="test", transformtype="test")

train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=batch_size,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, 
                                         batch_size=batch_size,
                                         num_workers=4)
test_loader = torch.utils.data.DataLoader(predict_set, batch_size=1, shuffle=False, num_workers=0)


# In[ ]:


# Hyperparameters
n_output=1
# Number of Epochs
num_epochs = 20
# Learning Rate
learning_rate = 0.0001
# Model parameters

# Show stats for every n number of batches
show_every_n_batches = 1


# In[ ]:


model_transfer = models.densenet121(pretrained=True)
print(model_transfer)


# In[ ]:


#model_transfer.features.conv0


# In[ ]:


def train_rnn(model, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    train_loss = 0
    valid_loss = 0
    valid_corrects = 0
    batch_losses = []
    val_batch_losses = []
    valid_loss_min = np.Inf
    
    model.train()
    
    previousLoss = np.Inf
    minLoss = np.Inf

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            #print(batch_idx + len(train_loader.dataset))
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            target = np.squeeze(target)
            #target = target.view(1,-1)
            target = target.long()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass: Train compute predicted outputs by passing inputs to the model
            
            output = model(data)
            
            #with torch.no_grad():
                #output = model(data)
        
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)
            
        model.eval()
        print("Model eval")
        for batch_idx, (data, target) in enumerate(val_loader):
            # move to GPU
            #print(batch_idx)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            target = np.squeeze(target)
            target= target.long()
            # zero the parameter gradients
            #optimizer.zero_grad()
            with torch.no_grad():
                output = model(data)
                _, preds = torch.max(output, 1)
                loss = criterion(output, target)
            #print(output)
            # update average validation loss 
            valid_loss += loss.item() * data.size(0)
            #print(loss.item())
            valid_corrects += torch.sum(preds == target.data)
        
        valid_loss = valid_loss/len(val_loader.dataset)
        train_loss = train_loss/len(train_loader.dataset)
        valid_corrects = valid_corrects.double()/len(val_loader.dataset)
        
        # print training/validation statistics 
        if epoch_i%show_every_n_batches == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch_i, train_loss, valid_loss))
        
            ## TODO: save the model if validation loss has decreased
            # save model if validation loss has decreased
            if valid_loss < valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Accuracy ({:.3}) Saving model ...'.format(
                valid_loss_min,
                valid_loss, valid_corrects))
                with open('trained_rnn_new', 'wb') as pickle_file:
                    torch.save(model.state_dict(), 'trained_rnn_new')
                valid_loss_min = valid_loss
                train_loss = 0
                valid_loss = 0
                valid_corrects = 0
                #batch_losses = []
                #val_batch_losses = []
  
    return model


# In[ ]:


# Freeze training for all "features" layers
# 362 features
count = 0
for param in model_transfer.features.parameters():
    count += 1
    if count < 100:
        param.requires_grad = False

## Freezing all layers
#for params in model_transfer.parameters():
#    params.requires_grad = False

#custom_model = nn.Linear(1024, 1)
custom_model = nn.Sequential(nn.Linear(1024, 1),nn.Softmax(dim=1))

"""
custom_model = nn.Sequential(nn.Linear(1024, 256), 
                  nn.ReLU(),
                  nn.Dropout(p=0.25), 
                  nn.Linear(256, 102)
                 )
"""
#if use_cuda:
#    custom_model = custom_model.to(device)
    
model_transfer.fc = custom_model

#model_transfer.load_state_dict(torch.load('save/trained_rnn_new'))

if use_cuda:
    model_transfer = model_transfer.cuda()
    model_transfer = torch.nn.DataParallel(model_transfer)

# print(model_transfer)

# specify loss function
criterion_scratch = nn.CrossEntropyLoss()
#criterion_scratch = nn.BCELoss()
#criterion_scratch = nn.BCELoss(reduction="mean").to('cuda:0')

# specify optimizer

#optimizer_scratch = optim.Adadelta(model_transfer.parameters())
optimizer_scratch = optim.SGD(model_transfer.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.000001)
#optimizer_scratch = optim.Adam(model_transfer.parameters(), lr=learning_rate)
#optimizer_scratch = optim.SGD(list(filter(lambda p: p.requires_grad, model_transfer.parameters())), lr=learning_rate, momentum=0.9)


#trained_rnn = train_rnn(model_transfer, batch_size, optimizer_scratch, criterion_scratch, num_epochs, show_every_n_batches)


# In[ ]:


model_transfer.load_state_dict(torch.load('trained_rnn_new'))
model_transfer.eval()
print("Model")


# In[ ]:


accuracy = 0
model_transfer.eval()
count = 0
equality = 0
for batch_idx, (data, target) in enumerate(val_loader):
    # move to GPU
    #print(batch_idx)
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    target = np.squeeze(target)
    target= target.long()
    # zero the parameter gradients
    #optimizer.zero_grad()
    with torch.no_grad():
        output = model_transfer(data)
        _, preds = torch.max(output, 1)
    #print(output)
    #print(target)
    equality = r2_score(preds.cpu(), target.cpu())
    if equality < 0:
        equality = 0
    
    #print(equality)
    count += 1
    # Accuracy is number of correct predictions divided by all predictions
    accuracy += equality
    
#print(count)
print("Test accuracy: {:.3f}".format(accuracy/len(val_loader)))


# In[ ]:


test_preds = []
submission = df_test
model_transfer.eval()
for batch_idx, (data, target) in enumerate(test_loader):
    # move to GPU
    if use_cuda:
        data = data.cuda()
        
    output = model_transfer(data)
    output = output.detach().cpu().max(1)[1]
    output = output.numpy()
    #print(output)
    test_preds = np.concatenate((test_preds, output), axis=0).astype(int)

print(test_preds)    
print(len(test_preds))
print(submission.shape)
submission["Class"] = test_preds
display(submission.head())


# In[ ]:


labels_dict_out = {0:"Large", 1:"Small"}
submission["Class"] = submission["Class"].apply(lambda x: labels_dict_out[x])
display(submission.head())


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename="submission_1.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe
create_download_link(submission)


# In[ ]:


get_ipython().run_line_magic('reset', '')

