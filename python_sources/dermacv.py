# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# To do:
    
#     1. lr_scheduler
#     2. transform
#     3. testing
#     4. visualize training at end
#     5. balance class quantities


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import torchvision.transforms.functional as TF

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        continue

#### Create Data Loaders
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch

from torchvision import transforms

# Image transformations
train_transforms = {
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ])}



class TrainData(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
        self.ids = self.df['image_id']
        self.dxs = self.df['dx']
        self.file_names = self.df['lesion_id']
        self.img_folder_path_1 = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1'
        self.img_folder_path_2 = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2'
        
        num_unique = self.df['dx'].unique()
        self.classes = len(num_unique)
        dx_decoder = {}
        for i, dx in enumerate(num_unique):
            dx_decoder[dx] = i
        self.dx_decoder = dx_decoder
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):

        file_sorter = self.ids.iloc[idx]
        last_five = int(file_sorter[-5:])
        file_name = self.file_names.iloc[idx]
        
        if last_five <= 29305:
            image_dir = self.img_folder_path_1 + '/' + file_sorter + '.jpg'
        else:
            image_dir = self.img_folder_path_2 + '/' + file_sorter + '.jpg'
        
        image = Image.open(image_dir)
        
        if self.transform is not None:
            x = self.transform(image)
        else:
            x = TF.to_tensor(image)
            
        dx_str = self.dxs[idx]      
        return x, self.dx_decoder[dx_str]

data_frame = pd.read_csv('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
train_set = TrainData(data_frame)#, transform = train_transforms)

train_val_ratio = 0.25
train_test_ratio = 0.15
np.random.seed(24)

num_train, _ = data_frame.shape
print(num_train)
indices = list(range(num_train))
np.random.shuffle(indices)
test_split = int(np.floor(train_test_ratio*num_train))
train_index, test_index = indices[test_split:], indices[:test_split]


num_val = len(train_index)
val_split = int(np.floor(train_val_ratio*num_val))
train_index, val_index = train_index[val_split:], train_index[:val_split]

train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(val_index)
test_sampler = SubsetRandomSampler(test_index)

batch_size = 16      #### BATCH_SIZE

train_loader = DataLoader(train_set, 
                          batch_size = batch_size, 
                          sampler = train_sampler,
                          num_workers = 16)

valid_loader = DataLoader(train_set, 
                          batch_size = batch_size, 
                          sampler = valid_sampler,
                          num_workers = 16)

test_loader = DataLoader(train_set, batch_size = batch_size, sampler = test_sampler, num_workers = 16)


############# Define network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.classifier[6] = nn.Sequential(
                  nn.Linear(4096, 256), 
                  nn.ReLU(), 
                  nn.Dropout(0.4),
                  nn.Linear(256, 7),                   
                  nn.LogSoftmax(dim=1))

# Move to gpu
model = model.to('cuda')
# Distribute across 2 gpus
model = nn.DataParallel(model)

# import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from torch import optim

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

print(torch.cuda.is_available())
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
lr_scheduler = ReduceLROnPlateau(optimizer, patience = 3, verbose = True)


epochs = 20
training_accuracies = []
validation_accuracies = []
training_losses = []
validation_losses = []

for epoch in range(epochs):
    
    training_accuracy = []
    training_loss = []
    model.train()
    for batch_num, training_batch in enumerate(train_loader):
        inputs, labels = training_batch
        optimizer.zero_grad()
        forward_output = model(inputs.float().to('cuda'))
        loss = loss_function(forward_output, labels.to('cuda'))
        forward_output = forward_output.data.cpu().numpy()
        forward_output = np.argmax(forward_output, axis = 1)
        training_accuracy.append(accuracy_score(labels.numpy(), forward_output))
        training_loss.append(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()
    training_losses.append(np.mean(training_loss))
    training_accuracies.append(np.mean(training_accuracy))    
        
    model.eval()
    with torch.no_grad():
        print("NOW EVALUATING")
        validation_accuracy = []
        validation_loss = []
        for batch_num, validation_batch in enumerate(valid_loader):
            inputs, actual_val = validation_batch
            predicted_val = model(inputs.float().to('cuda'))
            val_loss = loss_function(predicted_val, actual_val.to('cuda'))
            predicted_val = predicted_val.data.cpu().numpy()
            predicted_val = np.argmax(predicted_val, axis = 1)
            validation_accuracy.append(accuracy_score(actual_val.numpy(), predicted_val))
            validation_loss.append(val_loss.detach().cpu().item())
        validation_losses.append(np.mean(validation_loss))
        validation_accuracies.append(np.mean(validation_accuracy))

    print(f'Epoch {epoch}, Training loss {training_losses[-1] : 0.2E}, Training accuracy {training_accuracies[-1] : 0.2E},  Validation loss {validation_losses[-1] : 0.2E} Validation accuracy {validation_accuracies[-1] : 0.3f}, LR {optimizer.param_groups[0]["lr"] : 0.2E}')
    if lr_scheduler is not None:
      lr_scheduler.step(validation_losses[-1])

















