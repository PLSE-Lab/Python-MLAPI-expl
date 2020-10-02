#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General libraries imports
import pandas as pd
import numpy as np
import os
import time

# import image manipulation
from PIL import Image

# import matplotlib for visualization
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# Import PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader


# # Arial Cactus Identification with PyTorch and VGG16

# ## Introduction
# 
# The goal of the [Aerial Cactus Identification competition](https://www.kaggle.com/c/aerial-cactus-identification) is to create an algorithm that can identify a specific type of cactus in aerial imagery to advance a system for autonomous surveillance of protected areas.
# 
# <br>In this competition, we are given a dataset with labeled 32 x 32 images, which contain aerial photos of a columnar cactus. The task is to build an algorithm, which predicts whether there is a cactus on the image. This is a classification problem. Since we have a lot of data, deep learning models will suit well for this problem.
# 
# <br>In this analysis I used the code from this [Kaggle kernel](https://www.kaggle.com/atrisaxena/pytorch-simple-model-iscactus-classification).

# ## Explore Data
# 
# Let's explore the dataset given:
# * How many records are there in train and test datasets?
# * How many images are there with/without cactus?
# * Visualize images with and without cactus.

# In[ ]:


# Set path to folders containing the data
TRAIN_IMG_PATH = "../input/train/train/"
TEST_IMG_PATH = "../input/test/test/"
LABELS_CSV_PATH = "../input/train.csv"
SAMPLE_SUB_PATH = "../input/sample_submission.csv"


# `1` Number of images in train and test datasets:

# In[ ]:


# read the csv with labels for train dataset
pd_train = pd.read_csv(LABELS_CSV_PATH)

# count the number or rows
train_images = len(pd_train)

print("The number of images in train dataset is {}.".format(train_images))


# In[ ]:


# count the number of images in test dataset
test_images = len([f for f in os.listdir(TEST_IMG_PATH) if os.path.isfile(os.path.join(TEST_IMG_PATH, f))])

print("The number of images in test dataset is {}.".format(test_images))


# In[ ]:


# Plot pie chart
labels = 'Train', 'Test'
sizes = [train_images, test_images]
explode = (0, 0.1)  # "explode" the 2nd slice

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Number of images in training dataset")
plt.show()


# `2` Number of images with and without cactus in train dataset:

# In[ ]:


# calculate the number of images with and without cactus
with_cactus_num = pd_train[pd_train['has_cactus'] == 1].has_cactus.count()
no_cactus_num = pd_train[pd_train['has_cactus'] == 0].has_cactus.count()

print("The number of images with cactus in train dataset is {}.".format(with_cactus_num))
print("The number of images without cactus in train dataset is {}.".format(no_cactus_num))


# In[ ]:


# Plot pie chart
labels = 'Cactus', 'No cactus'
sizes = [with_cactus_num, no_cactus_num]
explode = (0, 0.1)  # "explode" the 2nd slice

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Number of images with/without cactus")
plt.show()


# `3` View examples of images with cactus and without cactus:

# In[ ]:


# get array of image filenames with and without cactus
has_cactus = pd_train[pd_train['has_cactus'] == 1][:9].id.values
no_cactus = pd_train[pd_train['has_cactus'] == 0][:9].id.values


# In[ ]:


def view_cactus(cactus_images, title = ''):
    """
    Function to plot grid with several examples of images.
    INPUT:
        cactus_images - array with filenames for images

    OUTPUT: None
    """
    fig, axs = plt.subplots(3, 3, figsize=(7,7))
    
    for im in range(0,9):
        # open image
        image = Image.open(os.path.join(TRAIN_IMG_PATH,cactus_images[im]))
        i = im // 3
        j = im % 3
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')

    # set suptitle
    plt.suptitle(title)
    plt.show()


# In[ ]:


view_cactus(has_cactus, title = 'Images with cactus')


# In[ ]:


view_cactus(no_cactus, title = 'Images without cactus')


# ## Modelling

# I decided to use VGG-16 pretrained model (see [this article](https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5) for further reading) to classify the images.

# `1` Load the dataset:

# In[ ]:


# Define the dataset
class CactusDataset(Dataset):
    """Cactus identification dataset."""

    def __init__(self, img_dir, dataframe, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.        
            dataframe (pandas.core.frame.DataFrame): Pandas dataframe obtained
                by read_csv().
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.id[idx]) 
        image = Image.open(img_name)
        label = self.labels_frame.has_cactus[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label] 
    
    
# define train transformations 
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
# define test transformations 
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# In[ ]:


dframe = pd.read_csv(LABELS_CSV_PATH)
cut = int(len(dframe)*0.95)
train, test = np.split(dframe, [cut], axis=0)
test = test.reset_index(drop=True)

train_ds = CactusDataset(TRAIN_IMG_PATH, train, train_transforms)
test_ds = CactusDataset(TRAIN_IMG_PATH, test, test_transforms)
datasets = {"train": train_ds, "val": test_ds}


# In[ ]:


trainloader = DataLoader(train_ds, batch_size=32,
                        shuffle=True, num_workers=0)

testloader = DataLoader(test_ds, batch_size=4,
                        shuffle=True, num_workers=0)


# `2` Setup hyperparameters:

# _Don't forget to enable GPU in kernel settings to run the model on GPU._

# In[ ]:


epochs = 20
batch_size = 128
learning_rate = 0.003
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
device


# `3` Load pretrained VGG-16 model:

# _Don't forget to enable Internet in kernel settings to download the model._

# In[ ]:


model = models.vgg16(pretrained=True)
model


# `4` Train the model:

# In[ ]:


# freeze parameters
for param in model.parameters():
    param.requires_grad = False

# add layers to train
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 12000)),
                          ('dr1', nn.Dropout(p = 0.3)),
                          ('bn1', nn.BatchNorm1d(num_features=12000)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(12000, 1000)),
                          ('dr2', nn.Dropout(p = 0.3)),
                          ('bn2', nn.BatchNorm1d(num_features=1000)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim = 1))
                          ]))
    
model.classifier = classifier


# In[ ]:


# set loss function
criterion = nn.NLLLoss()

# set optimizer, only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


# In[ ]:


# train the model
model.to(device)

steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()


# `5` Create submission file:

# In[ ]:


submission_df = pd.read_csv(SAMPLE_SUB_PATH)
output_df = pd.DataFrame(index=submission_df.index, columns=submission_df.keys() )
output_df['id'] = submission_df['id']
submission_df['target'] =  [0] * len(submission_df)

tdata_transform = transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

submission_ds = CactusDataset(TEST_IMG_PATH, submission_df, tdata_transform)

sub_loader = DataLoader(submission_ds, batch_size=1,
                        shuffle=False, num_workers=0)


def test_sumission(model):
    since = time.time()
    sub_outputs = []
    model.train(False)  # Set model to evaluate mode
    # Iterate over data.
    prediction = []
    for data in sub_loader:
        # get the inputs
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        prediction.append(int(pred))
      
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return prediction


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['has_cactus'] = test_sumission(model)
sub.to_csv('submission1.csv', index= False)

sub.head()

