#!/usr/bin/env python
# coding: utf-8

# # Analytics Vidhya - Game of Deep Learning || Computer Vision Hackathon
#  - Competition Link: https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/

# # Problem Statement
# Ship or vessel detection has a wide range of applications, in the areas of maritime safety, fisheries management, marine pollution, defence and maritime security, protection from piracy, illegal migration, etc.
# 
# Keeping this in mind, a Governmental Maritime and Coastguard Agency is planning to deploy a computer vision based automated system to identify ship type only from the images taken by the survey boats. You have been hired as a consultant to build an efficient model for this project.

# There are 5 classes of ships to be detected which are as follows:
# 
# * Cargo
# * Military
# * Carrier
# * Cruise
# * Tankers
# 
# 
# ![](https://datahack.analyticsvidhya.com/media/__sized__/contest_cover/god_2-thumbnail-1200x1200-90.jpg)

# # Approach
#  - Created a Custom Pytorch Data loader function
#  - Used Pytorch Pretrained Models
#  - Used Resnet50, Resnet152 and Resnet101
#  - Final Submission is result of votings based on 3 submission files created from 3 different models:
#      - Image size 224x224
#      - Data Augmenation
#      - Stratified Sampling split for training and validation data because of data imbalance
#      - Pre-trained network(resnet50, Resnet152 and Resnet101) - Unfreezed All Layers
#      - Saved the best model with good accuracy and used that to make predictions

# # Import Libraries

# In[ ]:


import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import pandas as pd
import os
from PIL import Image
import shutil
import torchvision
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
import cv2
import random


# In[ ]:


#define the transformations
transform_ship = transforms.Compose([transforms.ToTensor()])


# In[ ]:


SEED = 200
base_dir = '../input/'

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)


# # Custom class for loading data
# 

# In[ ]:


#read the train csv

train_data = pd.read_csv(base_dir+'/train/train.csv')


# In[ ]:


train_data.head()


# In[ ]:


### map imge names to labels
map_img_class_dict = {k:v for k, v in zip(train_data.image, train_data.category)}


# In[ ]:


#read the test csv

test_data = pd.read_csv(base_dir+'/test_ApKoW4T.csv')
test_data.head()


# In[ ]:


from PIL import Image

class ShipDataLoader(torch.utils.data.DataLoader):
    def __init__(self, CSVfolder, process='train', transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor()]), imgFolder='../input/train/images/',labelsDict = {}, y_labels = list(train_data.category)):
        
        self.process = process
        self.imgFolder = imgFolder
        self.CSVfolder = CSVfolder
        self.y = y_labels
        self.FileList = pd.read_csv(self.CSVfolder)['image'].tolist()
        self.transform = transform
        self.labelsDict = labelsDict
        
        if self.process =='train':
            self.labels = [labelsDict[i] for i in self.FileList]
        else:
            self.labels = [0 for i in range(len(self.FileList))]

    def __len__(self):
        return len(self.FileList)
    
    def __getitem__(self,idx):
        file_name =  self.FileList[idx]
        image_data=self.pil_loader(self.imgFolder+"/"+file_name)
        
        if self.transform:
            image_data = self.transform(image_data)
        
        if self.process == 'train':
            label = self.y[idx]
        else:
            label = file_name
            
        return image_data, label
    
    def pil_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


# In[ ]:


#define the batchsize
training_batchsize = 5


# In[ ]:


#retrieve the full data
full_data = ShipDataLoader(base_dir+'/train/train.csv',process = "train", imgFolder = base_dir+"/train/images", labelsDict = map_img_class_dict)


# In[ ]:


#create a dataloader
trainfull_loader = torch.utils.data.DataLoader(full_data, batch_size=training_batchsize, shuffle=True)


# # Visualization of Data

# In[ ]:


# dictionary ship encoding 
ship = {1: 'Cargo', 
        2: 'Military', 
        3: 'Carrier', 
        4: 'Cruise', 
        5: 'Tankers'}


# In[ ]:


#custom function to display images

def imshow(img, title):
    
    #convert image from tensor to numpy for visualization
    npimg = img.numpy()
    #define the size of a figure
    plt.figure(figsize = (15, 15))
    plt.axis("off")
    
    #interchaging the image sizes - transposing
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title, fontsize=15)
    plt.show()


# In[ ]:


#function to get images and feed into our custom function 'imshow'

def show_batch_images(dataloader):
    
    #getting the images
    images, labels = next(iter(dataloader))
    #make a grid from those images
    img = torchvision.utils.make_grid(images)
    imshow(img, "classes: " + str([str(x.item())+ " "+ ship[x.item()] for x in labels]))


# In[ ]:


show_batch_images(trainfull_loader)


# In[ ]:


show_batch_images(trainfull_loader)


# # Modeling

# In[ ]:


#checking for available gpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from torchvision import models
import torch.optim as optim

#define the number of classes for the final layer
num_classes = 5


# ### Data Augmentation
# 
# A common strategy for training neural networks is to introduce randomness in the input data itself. For example, you can randomly rotate, mirror, scale, and/or crop your images during training. This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.
# 
# To randomly rotate, scale and crop, then flip your images you would define your transforms like this:
# 
# ```python
# train_transforms = transforms.Compose([transforms.RandomRotation(30),
#                                        transforms.RandomResizedCrop(224),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.5, 0.5, 0.5], 
# ```
# 
# You'll also typically want to normalize images with `transforms.Normalize`. You pass in a list of means and list of standard deviations, then the color channels are normalized like so
# 
# ```input[channel] = (input[channel] - mean[channel]) / std[channel]```
# 
# Subtracting `mean` centers the data around zero and dividing by `std` squishes the values to be between -1 and 1. Normalizing helps keep the network work weights near zero which in turn makes backpropagation more stable. Without normalization, networks will tend to fail to learn.
# 
# You can find a list of all [the available transforms here](http://pytorch.org/docs/0.3.0/torchvision/transforms.html). When you're testing however, you'll want to use images that aren't altered (except you'll need to normalize the same way). So, for validation/test images, you'll typically just resize and crop.

# In[ ]:


Training_transforms = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.RandomHorizontalFlip(),
	transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1, hue=0.1),
	transforms.RandomAffine(degrees=15, translate=(0.3,0.3), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


# In[ ]:


validation_transforms = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])


# In[ ]:


from torch.utils.data.sampler import SubsetRandomSampler


# In[ ]:


#### 90-10 train-validation split
tr, val = train_test_split(train_data.category, stratify=train_data.category, test_size=0.15, random_state=10)

### Batchsize and parallelize

training_batchsize = 16
num_workers = 8

#### Idx for train and valid
train_sampler = SubsetRandomSampler(list(tr.index)) 
valid_sampler = SubsetRandomSampler(list(val.index))


# In[ ]:


len(list(tr.index))


# In[ ]:


len(list(val.index))


# In[ ]:


#### Train dataloader ####
traindataset = ShipDataLoader('../input/train/train.csv',"train", Training_transforms, '../input/train/images', map_img_class_dict)
train_loader = torch.utils.data.DataLoader(traindataset, batch_size= training_batchsize ,sampler=train_sampler,num_workers=num_workers)


# In[ ]:


show_batch_images(train_loader)


# In[ ]:


#### Valid dataloader ####

valdataset = ShipDataLoader('../input/train/train.csv',"train", validation_transforms, '../input/train/images', map_img_class_dict)
val_loader = torch.utils.data.DataLoader(valdataset, batch_size=training_batchsize,sampler=valid_sampler,num_workers=num_workers)


# In[ ]:


#### Test dataloader ####

testdataset = ShipDataLoader('../input/train/train.csv',"test", validation_transforms, '../input/train/images', map_img_class_dict)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=training_batchsize,num_workers=num_workers)


# In[ ]:


import torch.nn as nn
import copy


# In[ ]:


#create a iterator

dataiter = iter(train_loader)
images, labels = dataiter.next()

#shape of images bunch
print(images.shape)

#shape of first image in a group of 4
print(images[1].shape)

#class label for first image
print(labels[1])


# In[ ]:


num_classes


# In[ ]:


dataloaders = {"train": train_loader, "val": val_loader} 
dataset_sizes = {"train": len(list(tr.index)), "val": len(list(val.index))}


# In[ ]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device) - 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == "train":
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
            elif phase == "val":
                test_losses.append(epoch_loss)
                test_acc.append(epoch_acc)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    
    del inputs, labels
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #saving the best model
    torch.save(model.state_dict(best_model_wts),"saved.pth")
    print("best model saved")
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, test_losses, train_acc, test_acc


# In[ ]:


def make_predictions(dataloader, trained_model):

    pred = dict()
    trained_model.eval()

    for  data in dataloader:
        images, labels = data
        images = images.cuda()
        
        #push model to cuda
        trained_model.cuda()
        outputs = trained_model(images)
        
        #print(outputs)
        for i in range(len(images)):
            #print(torch.argmax(outputs[i]))
            detect_class = torch.argmax(outputs[i]).item() + 1
            pred[labels[i]] = detect_class   

    df = pd.DataFrame(list(pred.items()), columns=['image', 'category'])
    return(df)


# In[ ]:


import time


# # Resnet 50 Model

# In[ ]:


model_ft = models.resnet50(pretrained = True)

print(model_ft)


# In[ ]:


#number of trainable parameters in resent101 - before freezing

print("Number of trainable parameters: ", sum(p.numel() for p in model_ft.parameters() if p.requires_grad))


# In[ ]:


### Let's print the names of the layer stacks for our model
for name, child in model_ft.named_children():
    print(name)


# In[ ]:


model_ft.fc.in_features


# In[ ]:


# Parameters of newly constructed modules have requires_grad=True by default
fc = nn.Sequential(nn.Linear(model_ft.fc.in_features, 720),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(720, 256),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(64, 5),
                                nn.Softmax(dim = 1))


# In[ ]:


model_ft.fc = fc


# In[ ]:


print(model_ft)


# In[ ]:


criterion = nn.CrossEntropyLoss()

model_ft = model_ft.to(device)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0001, weight_decay=1e-3)

# Decay LR by a factor of 0.15 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.15)

scheduler_cosineAL = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, len(train_loader), eta_min=1e-6)


# # Training Resnet50 Model

# In[ ]:


model_trained, train_lr, test_lr,train_acc, test_acc = train_model(model_ft, criterion, optimizer_ft, scheduler_cosineAL, num_epochs=80)


# In[ ]:


#plot the losses

plt.plot(train_lr, label='Training loss')
plt.plot(test_lr, label='Validation loss')
plt.legend(frameon=False)
plt.show()


# In[ ]:


plt.plot(train_acc, label = "Training acc")
plt.plot(test_acc, label = "Validation acc")
plt.legend(frameon=False)
plt.show()


# # Making Predictions - Resnet50

# In[ ]:


test_df_res50 = make_predictions(test_loader, model_trained)


# In[ ]:


test_df_res50.rename(columns = {'category':'rs50_category'}, inplace = True) 
test_df_res50.head()


# In[ ]:


test_df_res50.rs50_category.value_counts()


# # Resnet 101 Model

# In[ ]:


model_ft = models.resnet101(pretrained = True)

print(model_ft)


# In[ ]:


#number of trainable parameters in resent101 - before freezing

print("Number of trainable parameters: ", sum(p.numel() for p in model_ft.parameters() if p.requires_grad))


# In[ ]:


### Let's print the names of the layer stacks for our model
for name, child in model_ft.named_children():
    print(name)


# In[ ]:


model_ft.fc.in_features


# In[ ]:


# Parameters of newly constructed modules have requires_grad=True by default
fc = nn.Sequential(nn.Linear(model_ft.fc.in_features, 720),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(720, 256),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(64, 5),
                                nn.Softmax(dim = 1))


# In[ ]:


model_ft.fc = fc


# In[ ]:


print(model_ft)


# In[ ]:


criterion = nn.CrossEntropyLoss()

model_ft = model_ft.to(device)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0001, weight_decay=1e-3)

scheduler_cosineAL = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, len(train_loader), eta_min=1e-6)


# # Training Resnet101 Model

# In[ ]:


model_trained, train_lr, test_lr,train_acc, test_acc = train_model(model_ft, criterion, optimizer_ft, scheduler_cosineAL, num_epochs=80)


# In[ ]:


#plot the losses

plt.plot(train_lr, label='Training loss')
plt.plot(test_lr, label='Validation loss')
plt.legend(frameon=False)
plt.show()


# In[ ]:


plt.plot(train_acc, label = "Training acc")
plt.plot(test_acc, label = "Validation acc")
plt.legend(frameon=False)
plt.show()


# # Making Predictions - Resnet101

# In[ ]:


test_df_res101 = make_predictions(test_loader, model_trained)


# In[ ]:


test_df_res101.rename(columns = {'category':'rs101_category'}, inplace = True) 

test_df_res101.head()


# In[ ]:


test_df_res101.rs101_category.value_counts()


# # resnet152 Model

# In[ ]:


model_ft = models.resnet152(pretrained = True)

print(model_ft)


# In[ ]:


#number of trainable parameters in resent101 - before freezing

print("Number of trainable parameters: ", sum(p.numel() for p in model_ft.parameters() if p.requires_grad))


# In[ ]:


### Let's print the names of the layer stacks for our model
for name, child in model_ft.named_children():
    print(name)


# In[ ]:


model_ft.fc.in_features


# In[ ]:


# Parameters of newly constructed modules have requires_grad=True by default
fc = nn.Sequential(nn.Linear(model_ft.fc.in_features, 720),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(720, 256),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(64, 5),
                                nn.Softmax(dim = 1))


# In[ ]:


model_ft.fc = fc


# In[ ]:


print(model_ft)


# In[ ]:


criterion = nn.CrossEntropyLoss()

model_ft = model_ft.to(device)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0001, weight_decay=1e-3)

scheduler_cosineAL = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, len(train_loader), eta_min=1e-6)


#  # Training resnet152 Model

# In[ ]:


model_trained, train_lr, test_lr,train_acc, test_acc = train_model(model_ft, criterion, optimizer_ft, scheduler_cosineAL, num_epochs=80)


# In[ ]:


#plot the losses

plt.plot(train_lr, label='Training loss')
plt.plot(test_lr, label='Validation loss')
plt.legend(frameon=False)
plt.show()


# In[ ]:


plt.plot(train_acc, label = "Training acc")
plt.plot(test_acc, label = "Validation acc")
plt.legend(frameon=False)
plt.show()


# # Making Predictions - resnet152

# In[ ]:


test_df_res152 = make_predictions(test_loader, model_trained)


# In[ ]:


test_df_res152.rename(columns = {'category':'res152_category'}, inplace = True) 

test_df_res152.head()


# In[ ]:


test_df_res152.res152_category.value_counts()


# # Merge Results

# In[ ]:


merge_results = [df.set_index(['image']) for df in [test_df_res50, test_df_res101, test_df_res152]]
merge_results = pd.concat(merge_results, axis=1).reset_index()


# In[ ]:


merge_results.head()


# In[ ]:


merged_pred = merge_results.drop("image", axis = 1)

#finding the most frequent value in a row
merged_pred = merged_pred.mode(axis = 1)[0]


# In[ ]:


final_df = pd.DataFrame({"image": list(merge_results.image)})
final_df["category"] = merged_pred


# In[ ]:


final_df.head()


# In[ ]:


final_df.to_csv("final_submission.csv")


# In[ ]:




