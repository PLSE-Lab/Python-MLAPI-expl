#!/usr/bin/env python
# coding: utf-8

# The goal of this problem is to find if two faces are related or not. My idea is to use VGG Face model to extract features of faces from images, and then find how close the second image's features are. This is a theoretical idea that I think could work so I've just given it a try.

# # Importing Neccessary Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import torchvision.transforms.functional as TF
import itertools
import torch.utils.data as data_utils


# In[ ]:


#Checking if CUDA is available or not
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# # Initializing VGG Face Model

# In[ ]:


class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38

def vgg_face_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

#Parkhi, Omkar M., Andrea Vedaldi, and Andrew Zisserman. "Deep Face Recognition." BMVC. Vol. 1. No. 3. 2015.


# In[ ]:


#Initializing Face Model
face_model = vgg_face_dag()
face_model
if train_on_gpu:
    face_model.cuda()


# # Checking available Training Data

# In[ ]:


print(os.listdir('../input'))
print(os.listdir('../input/train/F0765'))


# So the train data has folders of families and inside them are folders of people. The data of relations between images is given in file train_relationships.csv

# # Pre processing relationships data

# In[ ]:


df = pd.read_csv("../input/train_relationships.csv")
df.head()


# We can see the data has two columns of people that are related. Let's split these columns into three i.e. Family, Person1 and Person2 

# In[ ]:


new = df["p1"].str.split("/", n = 1, expand = True)

# making separate first name column from new data frame 
df["Family"]= new[0]
# making separate last name column from new data frame 
df["Person1"]= new[1]

# Dropping old Name columns
df.drop(columns =["p1"], inplace = True)

new = df["p2"].str.split("/", n = 1, expand = True)

# making separate first name column from new data frame 
df["Family2"]= new[0]
# making separate last name column from new data frame 
df["Person2"]= new[1]

# Dropping old Name columns
df.drop(columns =["p2"], inplace = True)
df.head()


# Family column is redundant so it's not needed

# In[ ]:


del df['Family2']


# We can also create some more data from Families for people that are not related. Then we can use this data to train our model on classes 'Related' and 'Not Related'. 'Not Related' instances can be numerous since there are many families that are not correlated. However, that would give a lot of data, We can create new data from people within families that are not related. Eg. A son and daughter would be related to their father and their mother, However, the mother and father won't be related themselves

# In[ ]:


#A new column in the existing dataframe with all values as 1, since these people are all related
df['Related'] = 1

#Creating a dictionary, and storing members of each family
df_dict = {}
for index, row in df.iterrows():
    if row['Family'] in df_dict:
        df_dict[row['Family']].append(row['Person1'])
    else:
        df_dict[row['Family']] = [row['Person1']]
        
#For each family in this dictionary, we'll first make pairs of people
#For each pair, we'll check if they're related in our existing Dataset
#If they're not in the dataframe, means we'll create a row with both persons and related value 0
i=1
for key in df_dict:
    pair = list(itertools.combinations(df_dict[key], 2))
    for item in pair:
        if len(df[(df['Family']==key)&(df['Person1']==item[0])&(df['Person2']==item[1])])==0         and len(df[(df['Family']==key)&(df['Person1']==item[1])&(df['Person2']==item[0])])==0:
            new = {'Family':key,'Person1':item[0],'Person2':item[1],'Related':0}
            df=df.append(new,ignore_index=True)
        
#Storing rows only where Person1 and Person2 are not same
df = df[(df['Person1']!=df['Person2'])]

#len(df[(df['Related']==1)])

print(df['Related'].value_counts())


# In[ ]:


#Checking Dataframe contents once
df.head()


# # Creating Custom Dataset

# We can load a custom dataset such that when our dataloader iterates, it iterates the index in dataframe, and returns a pair of people, with their corresponding class i.e. if they're 'Related' or 'Non Related'

# In[ ]:


class FamilyDataset(Dataset):
    """Family Dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.relations = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.relations)
    
    def __getpair__(self,idx):
        pair = self.root_dir+self.relations.iloc[idx,0] + '/' + self.relations.iloc[idx,1],        self.root_dir+self.relations.iloc[idx,0] + '/' + self.relations.iloc[idx,2]
        return pair
    
    def __getlabel__(self,idx):
        return self.relations.iloc[idx,3]
    
    def __getitem__(self, idx):
        pair =  self.__getpair__(idx)
        label = self.__getlabel__(idx)
        
        return idx,pair,label


# # Initializing Classification Model

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
                
        hidden_1 = 3500
        hidden_2 = 3000
        hidden_3 = 2500
        hidden_4 = 2000
        hidden_5 = 1600
        hidden_6 = 1200
        hidden_7 = 1000
        hidden_8 = 450
        hidden_9 = 200
        hidden_10 = 100
        hidden_11 = 50
        hidden_12 = 20
        
        output = 2
        
        self.fc1 = nn.Linear(2622, hidden_1, bias=True)
        
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, hidden_4)
        self.fc5 = nn.Linear(hidden_4, hidden_5)
        self.fc6 = nn.Linear(hidden_5, hidden_6)
        self.fc7 = nn.Linear(hidden_6, hidden_7)
        self.fc8 = nn.Linear(hidden_7, hidden_8)
        self.fc9 = nn.Linear(hidden_8, hidden_9)
        self.fc10 = nn.Linear(hidden_9, hidden_10)
        self.fc11 = nn.Linear(hidden_10, hidden_11)
        self.fc12 = nn.Linear(hidden_11, hidden_12)
        self.fc13 = nn.Linear(hidden_12, output)
        
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc8(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc9(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc10(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc11(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc12(x))
        x = self.dropout(x)
        
        x = self.fc13(x)
        return x


# In[ ]:


# TODO: Define transforms for the training data and testing data
transform = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_dataset= FamilyDataset(df=df,root_dir="../input/train/",transform=transform)


# In[ ]:


model = ClassificationNet()
if train_on_gpu:
    model.cuda()


# # Initializing Trainloader and Validloader

# In[ ]:


# number of subprocesses to use for data loading
num_workers = 8

# percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_dataset,sampler=valid_sampler,num_workers=num_workers)


# # Initializing Loss Function & Optimizer

# In[ ]:


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# # Training the model

# In[ ]:


# number of epochs to train the model
n_epochs = 2

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    valid_loss = 0.0
    train_class_correct = list(0 for i in range(2))
    train_class_total = list(0 for i in range(2))
    ###################
    # train the model #
    ###################
    model.train()
    batch=0
    v_batch=0
    for i,data, target in train_loader:
        train_loss = 0.0
        if os.path.exists(data[0][0]) and os.path.exists(data[1][0]):
            count=0
            for person1 in os.listdir(data[0][0]):
                for person2 in os.listdir(data[1][0]):

                    image1 = Image.open(data[0][0]+'/'+ person1)
                    x1 = TF.to_tensor(image1)
                    x1.unsqueeze_(0)

                    image2 = Image.open(data[1][0]+'/'+ person2)
                    x2 = TF.to_tensor(image2)
                    x2.unsqueeze_(0)

                    if train_on_gpu:
                        x1 = x1.cuda()
                        x2 = x2.cuda()

                    vgg1 = face_model.forward(x1)
                    vgg2 = face_model.forward(x2)

                    face_distance = vgg1-vgg2

                    if train_on_gpu:
                        target = target.cuda()
                        face_distance = face_distance.cuda()

                    optimizer.zero_grad()
                    output = model(face_distance)
                    _,pred = torch.max(output,1)
                    # calculate the loss
                    loss = criterion(output, target)
                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # perform a single optimization step (parameter update)
                    optimizer.step()
                    # update running training loss
                    
                    correct = pred.eq(target.view_as(pred))
                    
                    for i in range(len(target)):
                        label = target.data
                        train_class_correct[label] += correct.item()
                        train_class_total[label] += 1
                    
                    
                    train_loss += loss.item()*len(data)
                    count+=1
            batch+=1
            print('\r', 'Family', batch, 'Output', output, 'Training Loss: {:.6f}', train_loss/count, end='')
            
            if batch%30==0:
                for i in range(2):
                    if train_class_total[i] > 0:
                        print('\nTraining Accuracy of %5s: %2d%% (%2d/%2d)' % (
                            str(i), 100 * train_class_correct[i] / train_class_total[i],
                            np.sum(train_class_correct[i]), np.sum(train_class_total[i])))

                print('\nTraining Accuracy (Overall): %2d%% (%2d/%2d)' % (
                    100. * np.sum(train_class_correct) / np.sum(train_class_total),
                    np.sum(train_class_correct), np.sum(train_class_total)))       
     
#             ######################    
#             # validate the model #
#             ######################
            
    
    valid_class_correct = list(0 for i in range(2))
    valid_class_total = list(0 for i in range(2))
    model.eval() # prep model for evaluation
    for i,data, target in valid_loader:
        valid_loss = 0.0
        if os.path.exists(data[0][0]) and os.path.exists(data[1][0]):
            count=0
            for person1 in os.listdir(data[0][0]):
                for person2 in os.listdir(data[1][0]):
                    image1 = Image.open(data[0][0]+'/'+ person1)
                    x1 = TF.to_tensor(image1)
                    x1.unsqueeze_(0)

                    image2 = Image.open(data[1][0]+'/'+ person2)
                    x2 = TF.to_tensor(image2)
                    x2.unsqueeze_(0)

                    if train_on_gpu:
                        x1 = x1.cuda()
                        x2 = x2.cuda()

                    vgg1 = face_model.forward(x1)
                    vgg2 = face_model.forward(x2)

                    face_distance = vgg1-vgg2

                    if train_on_gpu:
                        target = target.cuda()
                        face_distance = face_distance.cuda()

                    output = model(face_distance)
                    # calculate the loss
                    loss = criterion(output, target)
                    valid_loss += loss.item()*len(data)
                    
                    #Check Predicted Class
                    _, pred = torch.max(output, 1)

                    #Compare predicted class to the correct class
                    correct = pred.eq(target.view_as(pred))
                    
                    for i in range(len(target)):
                        label = target.data
                        valid_class_correct[label]+= correct.item()
                        valid_class_total[label] += 1
        
                    count+=1

            # print training/validation statistics 
            # calculate average loss over an epoch
        v_batch+=1
        print('\r', 'Family', v_batch, 'Validation Loss: {:.6f}', valid_loss/count, end='')
        if v_batch%30==0:
            for i in range(2):
                if valid_class_total[i] > 0:
                    print('\nValidation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        str(i), 100 * valid_class_correct[i] / valid_class_total[i],
                        np.sum(valid_class_correct[i]), np.sum(valid_class_total[i])))

            print('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(valid_class_correct) / np.sum(valid_class_total),
                np.sum(valid_class_correct), np.sum(valid_class_total)))
            
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss


# Now that our model is trained, we'll see how the sample submissions are

# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.head()


# In[ ]:


new = sample_submission["img_pair"].str.split("-", n = 1, expand = True)

# making separate first name column from new data frame 
sample_submission["Person1"]= new[0]
# making separate last name column from new data frame 
sample_submission["Person2"]= new[1]

# Dropping old Name columns
sample_submission.head()


# In[ ]:


for i in range(len(sample_submission)):
    person1 = sample_submission.loc[i,'Person1']
    person2 = sample_submission.loc[i,'Person2']
    
    if os.path.exists('../input/test/'+person1) and os.path.exists('../input/test/'+person2):
        image1 = Image.open('../input/test/'+person1)
        x1 = TF.to_tensor(image1)
        x1.unsqueeze_(0)

        image2 = Image.open('../input/test/'+person2)
        x2 = TF.to_tensor(image2)
        x2.unsqueeze_(0)

        if train_on_gpu:
            x1 = x1.cuda()
            x2 = x2.cuda()

        vgg1 = face_model.forward(x1)
        vgg2 = face_model.forward(x2)

        face_distance = vgg1-vgg2
        output = model(face_distance)
        _,pred = torch.max(output,1)
        
        sample_submission.loc[i,'is_related'] = pred.item()


# In[ ]:


sample_submission.drop(columns =["Person1","Person2"], inplace = True)


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission['is_related'].value_counts()


# In[ ]:


output= sample_submission.to_csv('output.csv',index=False)


# # Helper Functions

# To download output and view images

# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[ ]:


create_download_link(sample_submission)


# In[ ]:


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

