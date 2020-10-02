#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Transfer learning is a popular method in computer vision because it allows for quickly building accurate models. With transfer learning, instead of starting the learning process from scratch, you leverage patterns that have been learned from pre-trained models as features extraction. A pre-trained model is a model that was trained on a large benchmark dataset, typically ImageNet, to solve a problem similar to the one that we want to solve - image classification.
# 
# Though less time than building from scratching, using transfer learning to train image classification models can still take a considerable amount of time. In my experience, models can take up to 30-45 minutes per epoch to train when involving large datasets and/or complicated models. This means the more epochs needed to improve the model, the longer overall runtime is needed. So how can we improve the training runtime when implementing transfer learning? We can precalculate the frozen layers then pass those values as the dataset into the training model.
# 
# Pre-trained models can typically be broken into two parts, a features and fully connected block. The features block is typically frozen and used as features extraction and the fully connected block is reconfigured for the specific dataset being applied and the respective number for classes to be predicted. During the forward and back-propagation steps of the training phase, the data passes through both blocks and the model is updated accordingly decreasing the model's loss function.
# 
# The process of passing all the data through both the feature and fully connected blocks can be improved by precalculcating the data values for the features block. With the calculated features data values, we then need to only iterate through the fully connected blocks to train our model. This kernal provides a walkthrough of how to implement such a strategy using PyTorch.

# ### Project Imports
# Let's start with importing modules needed for this project.

# In[ ]:


import numpy as np
import pandas as pd

# Porject modules
import os
print(os.listdir("../input"))
from collections import OrderedDict
import time

# PyTorch modules
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets


# ## Load Data
# Configure data augmentation requirements and load image data into memory.

# In[ ]:


# Useful project constants
BATCH_SIZE = 24
TYPES_OF_DATASETS = ['train', 'valid'] # Order matters her for train_model function


# In[ ]:


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../input/flower_data/flower_data/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in TYPES_OF_DATASETS}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=False)
              for x in TYPES_OF_DATASETS}
dataset_sizes = {x: len(image_datasets[x]) for x in TYPES_OF_DATASETS}
class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Configure Model
# Will be using densenet161 pretrained model for the kernal. For this model, the features and fully connected blocks are named "features" and "classifier" respectively.

# In[ ]:


model = models.densenet161(pretrained=True)
NUM_CLASSES = len(class_names)
IN_SIZE = model.classifier.in_features  # Expected in_features for desnsenet161 model
NUM_EPOCHS = 10

# Freeze feature block since we're using this model for feature extraction
for param in model.features.parameters():
    param.requires_grad = False
    
# Prep for model training
criterion = nn.CrossEntropyLoss()
# only classifier parameters are being optimized the rest are frozen
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# ## Define Model Training Functions

# In[ ]:


# Fit data to model
def fit(model, data_loader, criterion, optimizer=None, train=False):
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        if train:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
        # update running training loss
        running_loss += loss.item() * data.size(0)

        # Calculate accuracy
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        correct = top_class == target.view(*top_class.shape)
        running_acc += torch.mean(correct.type(torch.FloatTensor))
        
    loss = running_loss / len(data_loader.dataset)
    acc = 100. * running_acc / len(data_loader)
    
    return loss, acc


# In[ ]:


# Function for initiating model training
def train_model(model, train_loader, valid_loader, criterion, optimizer,
               scheduler, num_epochs=10):
    """
    Trains and validates the data using the specified model and parameters.
    """
    model.to(device)

    start_train_timer = time.time()
    for epoch in range(num_epochs):
        # Start timer
        start = time.time()
        # Pass forward through the model
        scheduler.step()
        train_loss, train_accuracy = fit(model, train_loader, criterion=criterion,
                                         optimizer=optimizer, train=True)
        valid_loss, valid_accuracy = fit(model, valid_loader, criterion=criterion,
                                         train=False)

        # calculate average loss over an epoch
        elapshed_epoch = time.time() - start

        # print training/validation statistics 
        print('Epoch: {} - completed in: {:.0f}m {:.0f}s'.format(
            epoch + 1, elapshed_epoch // 60, elapshed_epoch % 60))
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))
        print('\tTraining accuracy: {:.3f} \tValidation accuracy: {:.3f}'.format(
            train_accuracy, valid_accuracy))
            
    training_time = time.time() - start_train_timer
    hours = training_time // (60 * 60)
    training_time -= hours * 60 * 60
    print('Model training completed in: {:.0f}h {:.0f}m {:.0f}s'.format(
            hours, training_time // 60, training_time % 60))


# ## Benchmark Running Time
# Let's first begin by timing the typical transfer learning process where the data is repeated passed through both the features and full connected blocks. 

# In[ ]:


# Get dataloaders for training and validation sets
train_loader = dataloaders['train']
valid_loader = dataloaders['valid']


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Run training model without precalculating frozen features\ntrain_model(model, train_loader, valid_loader, criterion,\n            optimizer, scheduler, num_epochs=NUM_EPOCHS)')


# ## Precalculated Features Running Time

# In[ ]:


# Function for generating convoluted features and labels for given dataset and model
def preconvfeat(dataloader, model):
    model.to(device)
    conv_features = []
    labels_list = []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.features(inputs)  # calculate values for features block only
        conv_features.extend(output.data.cpu()) # save to CPU since it has much larger RAM
        labels_list.extend(labels.data.cpu())

    return (conv_features, labels_list)


# In[ ]:


# Convoluated feature dataset class for retrieving datasets
class ConvDataset(Dataset):
    def __init__(self, feats, labels):
        self.conv_feats = feats
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.conv_feats[idx], self.labels[idx]


# In[ ]:


# Custom fc class for densenet161 that uses precalculated feature values
class FullyConnectedModel(nn.Module):

    def __init__(self,in_size,out_size):
        super().__init__()
        self.fc = nn.Linear(in_size,out_size)

    def forward(self, features):
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        return out

fc = FullyConnectedModel(IN_SIZE, NUM_CLASSES)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Precalculate feature values for the training and validation data\nconv_train_data, labels_train_data = preconvfeat(dataloaders['train'], model)\nconv_valid_data, labels_valid_data = preconvfeat(dataloaders['valid'], model)\n# Convert the calculated data into Dataset opbject\nfeat_train_dataset = ConvDataset(conv_train_data, labels_train_data)\nfeat_valid_dataset = ConvDataset(conv_valid_data, labels_valid_data)\n# Generate DataLoader objects for calculated Datasets\ntrain_dataloader = DataLoader(feat_train_dataset, batch_size=BATCH_SIZE, shuffle=True)\nvalid_dataloader = DataLoader(feat_valid_dataset, batch_size=BATCH_SIZE, shuffle=True)\n\n# run training model\ntrain_model(fc, train_dataloader, valid_dataloader, criterion,\n            optimizer, scheduler, num_epochs=NUM_EPOCHS)")


# ## Summary
# 
# As you can see, precalculating the features block and iterating only over the fully connected block, even for a small datasets, has significant improvements in training run time. This process is valuable for working with larger datasets and models that require large number of epochs to train. Though memory constraints need to be evaluated when considering this process, especially for very large datasets, since the calculated features data can be very large and thus may exceed memory capacity.
