#!/usr/bin/env python
# coding: utf-8

# # Relative Humidity prediction using linear regression (minimal)
# 
# Using the boston housing dataset: https://archive.ics.uci.edu/ml/datasets/Air+Quality

# In[ ]:


# Uncomment and run the commands below if imports fail
get_ipython().system('conda install numpy pytorch torchvision cpuonly -c pytorch -y')
get_ipython().system('pip install matplotlib --upgrade --quiet')
get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


# Imports
import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import *
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split


# ## Read file 

# In[ ]:


dataframe = pd.read_csv('../input/airquality-uci/AirQualityUCI.csv', delimiter =';', decimal= ',')
# here you have to use delimiter ';' which will make your raw data into tabular form with colums and rows
# decimal = ',' makes your column values seperated with dot rather than comma
dataframe.head()
# , parse_dates = [['Date', 'Time']]


# # **Check Rows and Columns**

# In[ ]:


dataframe.shape


# # Remove Columns with missing values

# To drop rows or cos in csv, we use dropna()
# axis = 1 means columns 
# how = 'all' means if all column values are null, we will drop
# inplace = 'True' means that this drop() will affect the origianl dataframe.

# In[ ]:


drop_dataframe = dataframe.dropna(axis = 1, how = 'all', inplace = True)


# axis = 0 means index

# In[ ]:


drop_dataframe = dataframe.dropna(axis = 0, how = 'all')


# In[ ]:


drop_dataframe.tail()


# In[ ]:


drop_dataframe.isna() # check if there's null 
# return False if no null


# # Rename columns

# In[ ]:


drop_dataframe = drop_dataframe.rename(columns={'T': 'Temperature(\u2103)',
                                                'RH': 'Relative Humidity(%)',
                                                'AH': 'Absolute Humidity'})
drop_dataframe.head()
# For degree Celsius symbol, you can use the followings. Both work. 
# 'Temperature \N{DEGREE SIGN}C'
# 'Temperature(\u2103)'


# # Extract Date and Time

# In[ ]:


# however our date is in string format 
# extract date, month and year from date 
drop_dataframe['Day'] = pd.DatetimeIndex(drop_dataframe['Date']).day
drop_dataframe['month'] = pd.DatetimeIndex(drop_dataframe['Date']).month
drop_dataframe['year'] = pd.DatetimeIndex(drop_dataframe['Date']).year
drop_dataframe.head()


# In[ ]:


# Time is in string format in the given data.
# So we cannot usse datetime.hour to select it. 
# We will extract hour by the followings. 
drop_dataframe['Hour']=drop_dataframe['Time'].apply(lambda x: str(x)[0:2])
# what we want is float type and so convert string to float or int
drop_dataframe['Hour']= drop_dataframe['Hour'].apply(lambda x: float(x))
drop_dataframe.dtypes


# In[ ]:


# remove date and time columns
df = drop_dataframe.drop(['Date', 'Time'], axis = 1)
df.head()


# In[ ]:


df.shape


# In[ ]:


import os 
os.chdir(r'/kaggle/working')
df.to_csv(r'df.csv')


# # Select inputs and target

# In[ ]:


# To convert into Tensor, select columns in list form. If it is not numpy, you will see an error. 
input_cols = list(df.drop(['Relative Humidity(%)'], axis = 1)) 
#target_cols = list(df.columns)[11]
target_cols = list(df.drop(input_cols, axis = 1))
print('input_cols:', input_cols, '\n')
print('target_cols: ', target_cols)


# In[ ]:


# convert to numpy
input_arrays = drop_dataframe[input_cols].to_numpy()
target_arrays = drop_dataframe[target_cols].to_numpy()
input_arrays.shape, target_arrays.shape
#target_arrays


# In[ ]:


# convert input and targets into Pytorch Tensor, make sure data type is torch.float32
inputs = torch.from_numpy(np.array(input_arrays, dtype = 'float32'))
targets = torch.from_numpy(np.array(target_arrays, dtype = 'float32'))


# Next, we need to create PyTorch datasets & data loaders for training & validation. We'll start by creating a TensorDataset.

# # Prepare DataSet

# In[ ]:


dataset = TensorDataset(inputs, targets)


# To determine the fraction of data that will be used for creating the validation set, we will use 0.2. It means that you use 80% for training, 20 % for validation from the given data. Then use random_split to create training & validation datasets. 

# In[ ]:


val_percent = 0.2 
num_rows = df.shape[0]
num_cols = df.shape[1]
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset, [val_size, train_size])


# We will define batch size which is good for memory. If you train all data at the same time, your training will take a huge time.

# In[ ]:


batch_size = 64


# In[ ]:


train_loader = DataLoader(train_ds, batch_size, shuffle=True)
# need to shuffle the train_ds for better result
val_loader = DataLoader(val_ds, batch_size)
# validation dataset does not to be shuffled because it is just for fit and evaluation


# Check if it's working fine.

# In[ ]:


df.head()


# In[ ]:


for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break

# you won't see the same number in the same place as shown in above table because you shuffled the data 


# In[ ]:


target_arrays


# ## Create a linear Regression Model

# In[ ]:


input_size = len(input_cols)
output_size = len(target_cols)


# In[ ]:


class AQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size) # fill this (hint: use input_size & output_size defined above)
        
    def forward(self, xb):
        out = self.linear(xb)                          # fill this
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets)                      # fill this
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets)                          # fill this    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
            
model = AQModel()


# ## Training

# In[ ]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


# In[ ]:


result = evaluate(model, val_loader) # Before using model
print(result)


# In[ ]:


model = AQModel()
epochs = 10000
lr = 1e-8
history = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


losses = [r['val_loss'] for r in [result] + history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.title('val_loss vs. epochs');


# ## Prediction

# In[ ]:


def predict_single(x, model):
    xb = x.unsqueeze(0)
    return model(x).item()


# In[ ]:


x, target = val_ds[10]
pred = predict_single(x, model)
print("Input: ", x)
print("Target: ", target.item())
print("Prediction:", pred)


# ## Save and upload

# In[ ]:


torch.save(model.state_dict(), 'air-quality-linear-minimal.pth')


# In[ ]:


val_loss = [result] + history


# In[ ]:


jovian.log_metrics(val_loss=val_loss)


# In[ ]:


jovian.commit(project='air-quality-linear-minimal', environment=None, outputs=['air-quality-linear-minimal.pth'])
jovian.commit(project='air-quality-linear-minimal', environment=None, outputs=['air-quality-linear-minimal.pth']) # Kaggle commit fails sometimes, so try again..


# In[ ]:




