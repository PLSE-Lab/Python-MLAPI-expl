#!/usr/bin/env python
# coding: utf-8

# ## Pytorch Starter Pre-Trained Resnet50
# This kernel mostly implements the [Pytorch Transfer Learning tutorial](http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) with a custom dataset class and the resnet50 pretrained model from torchvision.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import time
from shutil import copyfile
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir, makedirs, getcwd, remove
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision
from torchvision import transforms, datasets, models


# ### Define Custom Dataset

# In[ ]:


class SeedlingDataset(Dataset):
    def __init__(self, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname).convert('RGB')
        labels = self.labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, int(labels)


# ### Define classes from directory structure

# In[ ]:


data_dir = '../input/plant-seedlings-classification/'
cache_dir = expanduser(join('~', '.torch'))

image_size = 224
batch_size = 4
classes = listdir(data_dir + 'train/')
classes = sorted(classes, key=lambda item: (int(item.partition(' ')[0])
                               if item[0].isdigit() else float('inf'), item))
num_to_class = dict(zip(range(len(classes)), classes))
num_to_class


# ### Copy torchvision model to temp directory where it is expected

# In[ ]:


if not exists(cache_dir):
    makedirs(cache_dir)

models_dir = cache_dir + '/' + 'models/'
if not exists(models_dir):
    makedirs(models_dir)

model_name = 'resnet50-19c8e357.pth'
src = '../input/pretrained-pytorch-models/' + model_name;
dest = models_dir + model_name
copyfile(src, dest)


# ### Create dataframe of training data

# In[ ]:


train = []
for index, label in enumerate(classes):
    path = data_dir + 'train/' + label + '/'
    for file in listdir(path):
        train.append(['{}/{}'.format(label, file), label, index])
    
df = pd.DataFrame(train, columns=['file', 'category', 'category_id',]) 
df


# ### Split training / validation data & 

# In[ ]:


train_data = df.sample(frac=0.7)
valid_data = df[~df['file'].isin(train_data['file'])]


# ### Prepare dataframe for predictions in submission format

# In[ ]:


sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')
sample_submission.columns = ['file', 'category']
sample_submission['category_id'] = 0
sample_submission


# ### Setup transforms, datasets, and dataloaders

# In[ ]:


train_trans = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_trans = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = SeedlingDataset(train_data, data_dir + 'train/', transform = train_trans)
valid_set = SeedlingDataset(valid_data, data_dir + 'train/', transform = valid_trans)
test_set = SeedlingDataset(sample_submission, data_dir + 'test/', transform = valid_trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

dataset_sizes = {
    'train': len(train_loader.dataset), 
    'valid': len(valid_loader.dataset)
}


# ### Define training method
# mostly if not entirely from pytorch transfer learning tutorial

# In[ ]:


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_batch = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                labels = labels.view(-1)
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                running_batch +=1

            epoch_loss = running_loss / running_batch
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ### Define the model
# Freeze pretrained layers
# add linear layer with number of classes

# In[ ]:


use_gpu = torch.cuda.is_available()

model = models.resnet50(pretrained=True)

#I recommend training with these layers unfrozen for a couple of epochs after the initial frozen training
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
if use_gpu:
    model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

loaders = {'train':train_loader, 'valid':valid_loader, 'test': test_loader}


# ### Train the model
# using one epoch due to time constraints

# In[ ]:


model = train_model(loaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=1)

