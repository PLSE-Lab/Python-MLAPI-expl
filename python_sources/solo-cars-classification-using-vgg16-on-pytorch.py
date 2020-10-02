#!/usr/bin/env python
# coding: utf-8

# This entry is for the Hackathon conducted on weekend 21-22nd July 2019 by a study group in the Secure & Private AI Scholarship Challenge. The goal is to identify the cars by their models. My idea is to use **Transfer Learning** from VGG-16 for this classification problem and see the best accuracy I can get.

# # Importing the neccessary libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io, transform
import torch.utils.data as data_utils


# # Checking folder structure

# In[ ]:


print(os.listdir("../input"))
print(os.listdir("../input/car_data/car_data"))
#print(os.listdir("../input/car_data/car_data/train"))


# So the folders are already split up by their classes. Train and Test folders are created already and we have to split train data into train and validation data manually.

# Now, we'll check the contents of the csv files provided. 

# In[ ]:


anno_train = pd.read_csv('../input/anno_train.csv',header=None)
print('-----Anno Train-----')
print(anno_train.head())

names = pd.read_csv('../input/names.csv',header=None)
print('-----Names-----')
print(names.head())

sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
print('-----Sample Submission-----')
print(sampleSubmission.head())


# It's not clear what the purpose of anno_train.csv is. names.csv contains names of all the classes and sample_submission.csv is just an output of the id of the image and it's corresponding predicted class.
# 
# For now, we can skip anno_train.csv and use names.csv.

# # Data Loading

# We can create a default dataset since the classes are already split here, but it's better to create a custom dataset so we can correspond our classes to the index. By default, train and test loaders will load by indices. We should check if folders(classes) listed in train and test are all listed in names.csv file.

# In[ ]:


for model in os.listdir("../input/car_data/car_data/train"):
    if names[names[0]==model].shape[0]>0:
        continue
    else:
        print(model + ' - not found in the file names.csv')


# The above output name is not present in the file. Let's check the file once

# In[ ]:


names


# At index 173, Ram C-V Cargo Van Minivan 2012 has been named as Ram C/V Cargo Van Minivan 2012. We have to handle this mismatch while loading the data in our dataloader

# In[ ]:


#The total number of classes
classes = names_df[0]


# # Creating a Custom Dataset
# We'll create a custom dataset that loads our class folders. However, for each class we'll return the index in the names file. This makes sure that our predictions and training is based on the labels in the dataframe.

# In[ ]:


class CarDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self,root,csv_file,transform=None):
        super(CarDataset,self).__init__(root,transform)
        self.df = pd.read_csv(csv_file,header=None)
        
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(CarDataset, self).__getitem__(index)
        # the complete image file path
        path = self.imgs[index][0]

        #Class name extracted from this complete path
        label = path.split('/')[5]
        
        #We replace the mismatched class name
        if label=='Ram C-V Cargo Van Minivan 2012':
            label='Ram C/V Cargo Van Minivan 2012'
            
        #The index in the dataframe of this particular class
        name_index = self.df.index[self.df[0]==label][0]
        
        #File name extracted from the complete path, this requirement is based on sample_submissions.csv
        file = path.split('/')[6][0:-4]
        tuple_with_path = (original_tuple + (file,) + (name_index,))
        return tuple_with_path


# # Creating Dataloaders

# In[ ]:


# percentage of training set to use as validation
valid_size = 0.2

# Transforms for the Training Data and Testing Data
train_transform = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_transform = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

train_data = CarDataset(root="../input/car_data/car_data/train",transform=train_transform,csv_file="../input/names.csv")
test_data = CarDataset(root="../input/car_data/car_data/test",transform=test_transform,csv_file="../input/names.csv")

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
    sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
    sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20,shuffle=True)


# # Initializing Pre-Trained model

# In[ ]:


# Load the pretrained model from pytorch
model = models.vgg16(pretrained=True)

# print out the model structure
print(model)


# We'll train our model using CUDA if it's available

# In[ ]:


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# We tune the pre trained model accoring to our preferences, We replace the last layer with a custom linear layer that outputs the number of classes we have 

# In[ ]:


# # Freeze training for all "features" layers
# for param in model.features.parameters():
#     param.requires_grad = False
    
n_inputs = model.classifier[6].in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))

model.classifier[6] = last_layer

# if GPU is available, move the model to GPU
if train_on_gpu:
    model.cuda()


# In[ ]:


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)


# # Model Training

# In[ ]:


# number of epochs to train the model
n_epochs = 10

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_loss = 0.0
    
    train_class_correct = list(0. for i in range(len(classes)))
    train_class_total = list(0. for i in range(len(classes)))
    
    ###################
    # train the model #
    ###################
    
    # model by default is set to train
    model.train()
    for batch_i, (data, index, path,target) in enumerate(train_loader):
        
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # calculate the batch loss
        loss = criterion(output, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # perform a single optimization step (parameter update)
        optimizer.step()
                
        # update training loss 
        train_loss += loss.item()
        
        #Check Predicted Class
        _, pred = torch.max(output, 1)
        
        #Compare predicted class to the correct class
        correct = pred.eq(target.view_as(pred))
         
        for i in range(len(target)):
            label = target.data
            train_class_correct[label] += correct.item()
            train_class_total[label] += 1
        
        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d Train Loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0
            
            for i in range(len(classes)):
                if valid_class_total[i] > 0:
                    print('Training Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        str(i), 100 * train_class_correct[i] / train_class_total[i],
                        np.sum(train_class_correct[i]), np.sum(train_class_total[i])))

            print('\Training Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(train_class_correct) / np.sum(train_class_total),
                np.sum(train_class_correct), np.sum(train_class_total)))
    
    valid_loss = 0.0
    
    valid_class_correct = list(0. for i in range(len(classes)))
    valid_class_total = list(0. for i in range(len(classes)))
    
    model.eval()
    for batch_i, (data, index, path,target) in enumerate(valid_loader):
        
        # move tensors to GPU if CUDA is available  
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # calculate the batch loss
        loss = criterion(output, target)
        
        # update validation loss 
        valid_loss += loss.item()
        
        #Check Predicted Class
        _, pred = torch.max(output, 1)
        
        #Compare predicted class to the correct class
        correct = pred.eq(target.view_as(pred))
        
        for i in range(len(target)):
            label = target.data
            valid_class_correct[label] += correct.item()
            valid_class_total[label] += 1
        
        if batch_i % 20 == 19:    # print validation loss every specified number of mini-batches
            print('Epoch %d, Batch %d Validation Loss: %.16f' %
                  (epoch, batch_i + 1, valid_loss / 20))
            valid_loss = 0.0
            
            for i in range(len(classes)):
                if valid_class_total[i] > 0:
                    print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        str(i), 100 * valid_class_correct[i] / valid_class_total[i],
                        np.sum(valid_class_correct[i]), np.sum(valid_class_total[i])))

            print('\Validation Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(valid_class_correct) / np.sum(valid_class_total),
                np.sum(valid_class_correct), np.sum(valid_class_total)))
    
    #We save our model if the validation loss decreases
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss


# We load the model with minimum Validation Loss

# In[ ]:


model.load_state_dict(torch.load('model.pt'))


# # Model Testing
# Since we don't have results of the test set we can't calculate Testing Loss and Testing Accuracy. So for now we just give our output for the test set
# 

# In[ ]:


#We create a custom dictionary to hold our results as required for submissions
results = {}

for batch_i, (data, index, path,target) in enumerate(train_loader):
    
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    
    prob = torch.nn.functional.softmax(output, dim=1)
    
    #Check our predicted class
    _, pred = torch.max(prob, 1)
    
    #Load this predicted class in our custom dictionary
    for k in range(len(pred)):
            name = path[k]
            results[name] = pred[k].cpu().tolist()


# We'll create a DataFrame from this custom dictionary. We'll check sample_submission.csv and prepare the output data according to it

# In[ ]:


sampleSubmission['Predicted'].unique()


# We see that sample submission has predicted a range of 1-196, whereas our model outputs and trains on 0-195, Hence we have to increase our outputs by 1

# In[ ]:


output = pd.DataFrame(results, index=[0]).transpose()
output = output.reset_index()
output.rename(columns={'index':'Id',0:'Predicted'}, inplace=True)
output['Predicted'] = output['Predicted']+1
output.head()


# # Download Link for Output

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

create_download_link(output)


# In[ ]:


output.to_csv('output.csv',index=False)


# # Training Accuracy and Validation Accuracy

# In[ ]:


for i in range(len(classes)):
    
    if valid_class_total[i] > 0:
        print('Training Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * train_class_correct[i] / train_class_total[i],
            np.sum(train_class_correct[i]), np.sum(train_class_total[i])))

    print('\Training Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(train_class_correct) / np.sum(train_class_total),
        np.sum(train_class_correct), np.sum(train_class_total)))

for i in range(len(classes)):
        if valid_class_total[i] > 0:
            print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * valid_class_correct[i] / valid_class_total[i],
                np.sum(valid_class_correct[i]), np.sum(valid_class_total[i])))

        print('\Validation Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(valid_class_correct) / np.sum(valid_class_total),
            np.sum(valid_class_correct), np.sum(valid_class_total)))

