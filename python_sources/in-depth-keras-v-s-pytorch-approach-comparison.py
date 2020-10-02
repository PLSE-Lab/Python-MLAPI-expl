#!/usr/bin/env python
# coding: utf-8

# # In Depth Keras v/s Pytorch Approach & Comparison

# ### Objective : 
# Main obejctive behind this notebook is to give an idea on building model with Keras and Pytorch. 
# 
# Starting from **Data Preparation, Creating custom DataLoaders, Model building, Validation of Model and Model comparison.**
# 
# I am trying to keep it as **simple** as i can so that newbie can also understand the workflow.
# 
# If you learn anything useful from this notebook then **Give Upvote :)

# * ## Contents of the Notebook:
# 
# ### Part1: Keras:
# #### 1) Data Preparation
# 
# #### 2) Network architecture training (MLP + CNN) 
# 
# #### 3) Evaluating model performance and testing
# 
# ### Part2: Pytorch:
# #### 1) Data Preparation
# 
# #### 2) Creating custom DataLoaders
# 
# #### 3) Build Pytoch model (MLP + CNN) and validating model performance 
# 
# ### Part3: Comparison between Keras and Pytorch
# #### 1) Compare performace of Keras and Pytorch (MLP)
# 
# #### 2) Compare performace of Keras and Pytorch (CNN)

# In[ ]:


# Importing basic modules

import os, sys
import numpy as np
import pandas as pd
import warnings, random
warnings.filterwarnings('ignore')
print(os.listdir('../input/fashionmnist'))


# In[ ]:


# Reading trainset and testset

train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train.shape, test.shape


# #### Since we have imported trainset as well as testset. Now next step is to transform data so that it will we can feed this data to Keras Model

# In[ ]:


import torch
from tensorflow import random as rd

seed = 123456789
rd.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ## Part1: Keras:
# ### 1) Data Preparation
# 

# In[ ]:


# Importing Keras modules

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Reshape
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical


# In[ ]:


train.head(1)


# In[ ]:


test.head(1)


# Let's divide dataset into train + label variables. 

# In[ ]:


train_label = train.label
train.drop(columns = 'label', inplace = True)

test_label = test.label
test.drop(columns = 'label', inplace = True)


# **Keras expects the labels to be Hot-One-Encoded form** Therefore we need to convert label into Hot-One-Encoding

# In[ ]:


train_label_keras = to_categorical(train_label, num_classes=10)
test_label_keras = to_categorical(test_label, num_classes=10)


# Also convert it to binary from greyscale image

# In[ ]:


train = train/255.0
test = test/255.0


# ### 2) Network architecture training - MLP

# ### => Multi-layer perceptron

# Now we are ready to build an architecture. We are going to build a 4-Hidden layer network with dropout.

# To save model best weights and bias we are going to use **ModelCheckpoint from keras callback module**. Furthermore we will utilise this checkpoint to load best parameters for training set and use it to check model performance on testset

# In[ ]:


import tensorflow as tf
seed = 2010
tf.random.set_seed(seed)
np.random.seed(seed)


# In[ ]:


# Model architecture

model_mlp = Sequential()
model_mlp.add(Dense(784, activation='relu', use_bias=True))
model_mlp.add(Dropout(0.3))
model_mlp.add(BatchNormalization())
model_mlp.add(Dense(256, activation='relu', use_bias=True))
model_mlp.add(Dropout(0.4))
model_mlp.add(BatchNormalization(momentum=0.4))
model_mlp.add(Dense(128, activation='relu', use_bias=True))
model_mlp.add(Dropout(0.3))
model_mlp.add(BatchNormalization())
model_mlp.add(Dense(64, activation='relu', use_bias=True))
model_mlp.add(Dropout(0.4))
model_mlp.add(BatchNormalization(momentum=0.4))
model_mlp.add(Dense(10, activation='softmax', use_bias=True))
model_mlp.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )
check = ModelCheckpoint('weight.best.hdf5', monitor ='val_loss', save_best_only=True)


# In[ ]:


# Also using learning Rate reduction when we are moving towards optimized weights and bias
lr = ReduceLROnPlateau(monitor='val_loss', patience=7, min_lr=0.0001)
#es = EarlyStopping(monitor='val_loss', patience=5)
model_mlp.fit(x = train.values, y = train_label_keras, epochs=15, batch_size=64, validation_split=0.20, callbacks=[lr,check])


# ### => CNN On Keras

# We will try to built a Convolutional network model and then we will compare different models

# In[ ]:


# Model architecture

model_cnn = Sequential()
model_cnn.add(Reshape((28,28,-1), input_shape = (784,)))
model_cnn.add(Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same' ))
model_cnn.add(Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same' ))
model_cnn.add(BatchNormalization())
model_cnn.add(MaxPooling2D(pool_size =(2,2), strides = (2,2)))
model_cnn.add(Dropout(0.3))

model_cnn.add(Conv2D(128, kernel_size=(3,3), strides =(1,1), padding = 'same' ))
model_cnn.add(Conv2D(256, kernel_size=(3,3), strides =(2,2), padding = 'same' ))
model_cnn.add(BatchNormalization())
model_cnn.add(MaxPooling2D(pool_size =(2,2), strides =(2,2)))
model_cnn.add(Dropout(0.3))

model_cnn.add(Flatten())
model_cnn.add(Dense(256, activation='relu', use_bias=True))
model_cnn.add(Dropout(0.3))

model_cnn.add(Dense(64, activation='relu', use_bias=True))
model_cnn.add(Dropout(0.3))
model_cnn.add(BatchNormalization())

model_cnn.add(Dense(10, activation='softmax', use_bias=True))
model_cnn.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )
check = ModelCheckpoint('weight_cnn.best.hdf5', monitor ='val_loss', save_best_only=True)

# Also using learning Rate reduction when we are moving towards optimized weights and bias
lr = ReduceLROnPlateau(monitor='val_loss', patience=7)
#es = EarlyStopping(monitor='val_loss', patience=5)
model_cnn.fit(x = train.values, y = train_label_keras, epochs=15, batch_size=64, validation_split=0.20, callbacks=[lr,check])


# ### 3) Evaluating model performance and testing

# let's check model performace and test it on testset for **MLP**

# In[ ]:


import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,2, figsize = (20,7))
plt.subplot(1,2,1)
plt.plot([a for a in range(1,16)], model_mlp.history.history['val_loss'])
plt.plot([a for a in range(1,16)], model_mlp.history.history['loss'])
plt.legend(['validation_loss', 'training_loss'])

plt.subplot(1,2,2)
plt.plot([a for a in range(1,16)], model_mlp.history.history['val_accuracy'])
plt.plot([a for a in range(1,16)], model_mlp.history.history['accuracy'])
plt.legend(['validation_accuracy', 'training_accuracy'])


# Checking performance on **CNN** below

# In[ ]:


import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,2, figsize = (20,7))
plt.subplot(1,2,1)
plt.plot([a for a in range(1,16)], model_cnn.history.history['val_loss'])
plt.plot([a for a in range(1,16)], model_cnn.history.history['loss'])
plt.legend(['validation_loss', 'training_loss'])

plt.subplot(1,2,2)
plt.plot([a for a in range(1,16)], model_cnn.history.history['val_accuracy'])
plt.plot([a for a in range(1,16)], model_cnn.history.history['accuracy'])
plt.legend(['validation_accuracy', 'training_accuracy'])


# #### Now evaluate performace on testset using model best weights and bias parameters

# In[ ]:


print("MLP Performace")
model_mlp.load_weights('weight.best.hdf5')
score_keras = model_mlp.evaluate(x= test, y = test_label_keras, batch_size=256)
print('Keras Accuracy on testset is :  ', score_keras[1])
print('Keras Loss on testset is     :  ', score_keras[0])


# In[ ]:


print("CNN Performace")
model_cnn.load_weights('weight_cnn.best.hdf5')
score_keras = model_cnn.evaluate(x= test, y = test_label_keras, batch_size=256)
print('Keras Accuracy on testset is :  ', score_keras[1])
print('Keras Loss on testset is     :  ', score_keras[0])


# ### On Keras we get an Accuracy of 73-79% wheras on CNN keras we got accuracy of 74-82%.Now let's check Pytorch

# ## Part2: Pytorch:
# ### 1) Data Preparation

# In[ ]:


import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import random


# #### First we need to make Dataloader to load testset and trainset. With dataloader we will train our model and validate it's performance on testset

# In[ ]:


# Custom Dataset

class MyData(Dataset):
  def __init__(self, data, label, transform = None):
    self.data = data
    self.label = label
  
  def __len__(self):
    return(len(self.data))

  def __getitem__(self,idx):
    self.x = self.data.loc[idx,:].values
    self.y = self.label.loc[idx]
    return(self.x, self.y)


# In[ ]:


# Dividing data into train and valid set

indices = len(train)
indices = [a for a in range(indices)]
split = 0.20
split = int(np.floor(split*len(train)))
random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


# We are ready to create DataLoader to for trainset, validset, testset

# In[ ]:


# Creating DataLoaders

training = MyData(train, train_label, transforms.ToTensor())
training_loader = DataLoader(training, batch_size=64, sampler=train_sampler)

valid_loader = DataLoader(training, batch_size=64, sampler=valid_sampler)

testing = MyData(test, test_label, transforms.ToTensor())
testing_loader = DataLoader(testing, batch_size=64)


# ### 2) Network architecture training

# We will now build network architecture and train it on trainset. Continuously we will be monitoring on validation data. 

# ### => MLP on Pytorch

# In[ ]:


pymodel_mlp = nn.Sequential(nn.Linear(784,784),
                        nn.ReLU(),
                        nn.Dropout(p=0.3),
                        nn.BatchNorm1d(784),
                        nn.Linear(784,256),
                        nn.ReLU(),
                        nn.Dropout(p=0.4),
                        nn.BatchNorm1d(256),
                        nn.Linear(256,128),
                        nn.ReLU(),
                        nn.Dropout(p=0.3),
                        nn.BatchNorm1d(128),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Dropout(p=0.4),
                        nn.BatchNorm1d(64),
                        nn.Linear(64,10),
                        nn.LogSoftmax(dim = 1),)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(pymodel_mlp.parameters(), lr = 0.008)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pymodel_mlp = pymodel_mlp.to(device)
epoch = 15
valid_score = np.inf
torch_valid_loss = []
torch_valid_acc  = []

for e in range(epoch):
  training_loss = 0
  valid_loss  = 0

  running_loss = 0
  pymodel_mlp.train()
  for image, label in training_loader :
    optimizer.zero_grad()
    image = image.to(device)
    label = label.to(device)
    out = pymodel_mlp(image.float())
    loss = criterion(out,label)
    running_loss += loss.item()
    loss.backward()
    optimizer.step()
  
  with torch.no_grad():
    running_valid = 0
    acc_valid = 0
    pymodel_mlp.eval()
    for image, label in valid_loader:
      image = image.to(device)
      label = label.to(device)
      out = pymodel_mlp(image.float())
      loss = criterion(out, label)
      top_p, top_class = torch.exp(out).topk(1, dim = 1)
      equal = top_class == label.view(top_class.shape)
      accuracy = torch.mean(equal.type(torch.FloatTensor))
      running_valid += loss.item()
      acc_valid += accuracy.item()
    
    if running_valid < valid_score :
      torch.save(pymodel_mlp.state_dict(), 'checkpoint.pth')
      print('\nError changes from {} to {}'.format(valid_score/len(valid_loader), running_valid/len(valid_loader)))
      valid_score = running_valid
      print('Saved model\n')
        
  training_loss = running_loss/len(training_loader)
  valid_loss    = running_valid/len(valid_loader)
    
  torch_valid_loss.append(valid_loss)
  torch_valid_acc.append(acc_valid/len(valid_loader))
  print('Epoch : %s\nTraining_error : %s\nValid_error    : %s\nAccuracy_valid : %s\n------------------------------' 
          %(e+1, training_loss, valid_loss, acc_valid/len(valid_loader)))


# ### => CNN on Pytorch

# In[ ]:


from torch.nn import functional as f
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_32 = nn.Conv2d(1,32,3,padding=1)
        self.conv2d_64 = nn.Conv2d(32,64,3,padding=1)
        self.max2d     = nn.MaxPool2d(2,2)
        self.conv2d_128 = nn.Conv2d(64,128,3,padding=1)
        self.conv2d_256 = nn.Conv2d(128,256,3, stride = 2,padding=1)
        self.linear1    = nn.Linear(3*3*256, 256)
        self.linear2    = nn.Linear(256,64)
        self.linear3    = nn.Linear(64,10)
        self.batch2d1     = nn.BatchNorm2d(64)
        self.batch2d2    = nn.BatchNorm2d(256)
        self.batch1d     = nn.BatchNorm1d(64)
        self.drop      = nn.Dropout(p=0.3)
        self.flat      = nn.Flatten()
    
    def forward(self,x):
        x = x.view(-1,1,28,28)
        x = f.relu(self.conv2d_32(x))
        x = f.relu(self.conv2d_64(x))
        x = self.batch2d1(x)
        x = f.relu(self.max2d(x))
        x = self.drop(x)
        
        x = f.relu(self.conv2d_128(x))
        x = f.relu(self.conv2d_256(x))
        x = self.batch2d2(x)
        x = f.relu(self.max2d(x))
        x = self.drop(x)
        
        x = self.flat(x)
        x = f.relu(self.linear1(x))
        x = self.drop(x)
        x = f.relu(self.linear2(x))
        x = self.drop(x)
        x = self.batch1d(x)
        x = f.log_softmax(self.linear3(x), dim=1)
        return(x)

pymodel_cnn = net()
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(pymodel_cnn.parameters(), lr = 0.008)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pymodel_cnn = pymodel_cnn.to(device)
epoch = 15
valid_score = np.inf
torch_valid_loss_cnn = []
torch_valid_acc_cnn  = []

for e in range(epoch):
  training_loss = 0
  valid_loss  = 0

  running_loss = 0
  pymodel_cnn.train()
  for image, label in training_loader :
    optimizer.zero_grad()
    image = image.to(device)
    label = label.to(device)
    out = pymodel_cnn(image.float())
    loss = criterion(out,label)
    running_loss += loss.item()
    loss.backward()
    optimizer.step()
  
  with torch.no_grad():
    running_valid = 0
    acc_valid = 0
    pymodel_cnn.eval()
    for image, label in valid_loader:
      image = image.to(device)
      label = label.to(device)
      out = pymodel_cnn(image.float())
      loss = criterion(out, label)
      top_p, top_class = torch.exp(out).topk(1, dim = 1)
      equal = top_class == label.view(top_class.shape)
      accuracy = torch.mean(equal.type(torch.FloatTensor))
      running_valid += loss.item()
      acc_valid += accuracy.item()
    
    if running_valid < valid_score :
      torch.save(pymodel_cnn.state_dict(), 'checkpoint_cnn.pth')
      print('\nError changes from {} to {}'.format(valid_score/len(valid_loader), running_valid/len(valid_loader)))
      valid_score = running_valid
      print('Saved model\n')
        
  training_loss = running_loss/len(training_loader)
  valid_loss    = running_valid/len(valid_loader)
    
  torch_valid_loss_cnn.append(valid_loss)
  torch_valid_acc_cnn.append(acc_valid/len(valid_loader))
  print('Epoch : %s\nTraining_error : %s\nValid_error    : %s\nAccuracy_valid : %s\n------------------------------' 
          %(e+1, training_loss, valid_loss, acc_valid/len(valid_loader)))


# ### 3) Evaluating Pytorch model performance and testing it on testset

# Loading best pytorch model on MLP and testing it on testset

# In[ ]:


state = torch.load('checkpoint.pth')
pymodel_mlp.load_state_dict(state)


# In[ ]:


torch_test_loss  = []
torch_test_acc   = []

for e in range(1):
    testing_loss = 0
    running_test = 0
    acc_test = 0
    pymodel_mlp.eval()
    for image, label in testing_loader:
      image = image.to(device)
      label = label.to(device)
      out = pymodel_mlp(image.float())
      loss = criterion(out, label)
      top_p, top_class = torch.exp(out).topk(1, dim = 1)
      equal = top_class == label.view(top_class.shape)
      accuracy = torch.mean(equal.type(torch.FloatTensor))
      running_test += loss.item()
      acc_test += accuracy.item()
    testing_loss = running_test/len(testing_loader)
    torch_test_loss.append(testing_loss)
    torch_test_acc.append(acc_test/len(testing_loader))
    print('Epoch : %s\nTesting_error : %s\nAccuracy_test : %s\n----------------' %(e+1, testing_loss, acc_test/len(testing_loader)))


# #### Here we got an accuracy of 87 on testset.

# Let's try on Trained CNN pytorch model

# In[ ]:


state = torch.load('checkpoint_cnn.pth')
pymodel_cnn.load_state_dict(state)

torch_test_loss_cnn  = []
torch_test_acc_cnn   = []

for e in range(1):
    testing_loss = 0
    running_test = 0
    acc_test = 0
    pymodel_cnn.eval()
    with torch.no_grad():
        for image, label in testing_loader:
          image = image.to(device)
          label = label.to(device)
          out = pymodel_cnn(image.float())
          loss = criterion(out, label)
          top_p, top_class = torch.exp(out).topk(1, dim = 1)
          equal = top_class == label.view(top_class.shape)
          accuracy = torch.mean(equal.type(torch.FloatTensor))
          running_test += loss.item()
          acc_test += accuracy.item()
    testing_loss = running_test/len(testing_loader)
    torch_test_loss_cnn.append(testing_loss)
    torch_test_acc_cnn.append(acc_test/len(testing_loader))
    print('Epoch : %s\nTesting_error : %s\nAccuracy_test : %s\n----------------' %(e+1, testing_loss, acc_test/len(testing_loader)))


# ### Pytorch CNN is performing better than Pytorch MLP model

# ### ALSO pytorch model is performing better than Keras model. Let's get into more detailed analysis 

# ## Part3: Comparison between Keras and Pytorch
# ### 1) Compare performace of Keras and Pytorch

# Let's compare Validation set loss and Accuracy between Pytorch and Keras

# In[ ]:


import matplotlib.pyplot as plt

fig,ax = plt.subplots(2,1, figsize = (18,12))
plt.subplot(2,1,1)
plt.title('Keras v/s Pytorch (MLP) Loss', fontsize = 20)
plt.plot([a for a in range(1,16)], torch_valid_loss)
plt.plot([a for a in range(1,16)], model_mlp.history.history['val_loss'])
plt.legend(['Pytorch Validation_loss', 'Keras Validation_loss'])

plt.subplot(2,1,2)
plt.title('Keras v/s Pytorch (MLP) Accuracy', fontsize = 20)
plt.plot([a for a in range(1,16)], torch_valid_acc)
plt.plot([a for a in range(1,16)], model_mlp.history.history['accuracy'])
plt.legend(['Pytorch Validation_Accuracy', 'Keras Validation_Accuracy'])


# In[ ]:


import matplotlib.pyplot as plt

fig,ax = plt.subplots(2,1, figsize = (18,12))
plt.subplot(2,1,1)
plt.title('Keras v/s Pytorch (CNN) Loss' , fontsize = 20)
plt.plot([a for a in range(1,16)], torch_valid_loss_cnn)
plt.plot([a for a in range(1,16)], model_cnn.history.history['val_loss'])
plt.legend(['Pytorch Validation_loss', 'Keras Validation_loss'])

plt.subplot(2,1,2)
plt.title('Keras v/s Pytorch (CNN) Accuracy', fontsize = 20)
plt.plot([a for a in range(1,16)], torch_valid_acc_cnn)
plt.plot([a for a in range(1,16)], model_cnn.history.history['accuracy'])
plt.legend(['Pytorch Validation_Accuracy', 'Keras Validation_Accuracy'])


# ### Pytorch model out performed Keras.

# ### Accuracy on testset better in Pytorch also we are mostly getting near ablut 80%-90% accuarcy with Pytorch model. Whereas on Keras we are getting accuracy of 70-80%.  

# ### Feel free to give suggestion/feedback. Upvote it if you like this
