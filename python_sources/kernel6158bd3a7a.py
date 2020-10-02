#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Refs:
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#from sklearn import svm
#import lightgbm as lgb
import time
import os
import math
import logging

from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader




        
        
        
# list of parameters: dev_ratio

    # hyperpapeters
    



# set up logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Prepare for the train and dev datasets

train = pd.read_csv('/kaggle/input/learn-together/train.csv')
ordered_train = train.sort_values(by = 'Cover_Type')
cover_type_dict = ordered_train['Cover_Type'].value_counts().to_dict()

for k, v in cover_type_dict.items():
    print(k, v)

X = ordered_train.drop(['Id', 'Cover_Type'], axis = 1)
y = ordered_train['Cover_Type'] - 1 # Cover_Type: 0--6


# Preprocess of the data (rescale the data based on the estimated mean and variance across all the features)
# apply to train, dev, and test datasets

scaler = preprocessing.StandardScaler().fit(X)
#X_transformed = scaler.transform(X)



# helper function for separate train and dev datasets

dev_ratio = 0.2

def split(features, labels, dev_ratio):
    '''inputs:
                features: features pandas array
                labelss: labels for the feature pandas array
                dev_ratio: the proportion of data that will be used in the dev sets
                
        outputs:
                a tuple of (features_train, labels_train, features_dev, labels_dev)
    '''
    if not isinstance(features, pd.DataFrame):
        raise TypeError('features and labels should be pandas dataframes')
    
    if features.shape[0] != labels.shape[0]:
        raise ValueError('features and labels are not compatiable')
        
        
    if dev_ratio <= 0 or dev_ratio >= 1:
        raise ValueError('illegal dev_ratio input, should be in (0,1)')
    
    dev_indeces = np.random.choice(np.arange(labels.shape[0]), 
                           size = math.floor(labels.shape[0]*dev_ratio), replace = None)
    
    train_indeces = np.delete(np.arange(labels.shape[0]), dev_indeces)
    
    logging.debug('{:.0f}% of data will be used for dev.'.format(dev_ratio*100))
    
    return X.iloc[train_indeces], y.iloc[train_indeces], X.iloc[dev_indeces], y.iloc[dev_indeces]

X_train, y_train, X_dev, y_dev = split(X, y, .2)

X_train_trans = scaler.transform(X_train)  
X_dev_trans = scaler.transform(X_dev) 
train_ds = TensorDataset(torch.tensor(X_train_trans, dtype = torch.float), 
                         torch.tensor(y_train.values))
dev_ds = TensorDataset(torch.tensor(X_dev_trans, dtype = torch.float), 
                            torch.tensor(y_dev.values))


# helper function for cross-validation



# helper function for MLP


# MLP model (use as a reference case for comparing with MLP+DenseNet)


lr = .01
momentum = 0.1
weight_decay = 0.001
epochs = 10
loss_func = F.cross_entropy
batch_accuracies = []

def get_data(train_ds, validate_ds, batch_size):
    return (
        DataLoader(train_ds, batch_size = batch_size, shuffle = True),
        DataLoader(dev_ds, batch_size = batch_size )
    )

mlp = nn.Sequential()
mlp.add_module('linear1', nn.Linear(54,100))
mlp.add_module('ReLU1', nn.ReLU())
mlp.add_module('linear2', nn.Linear(100,49))
mlp.add_module('ReLU2', nn.ReLU())
mlp.add_module('linear3', nn.Linear(49,7))
mlp.add_module('ReLU4', nn.ReLU())
mlp.add_module('linear4', nn.Linear(7,7))







def get_model(model, opt_method, **kwargs):
    if opt_method.lower() == 'sgd':
        try:
            opt = torch.optim.SGD(model.parameters(), **kwargs)
        except TypeError as e:
            print(e)
            raise
        else:
            return model, opt
        
    elif opt_method.lower() == 'adam':
        try:
            opt = torch.optim.Adam(model.parameters(), **kwargs)
        except TypeError as e:
            print(e)
            raise
        else:
            return model, opt
    else:
        raise ValueError('cannot find optimization method')




def loss_batch(model, loss_func, xb, yb, opt = None):
    out = model(xb)
    loss = loss_func(out, yb)
    preds = torch.argmax(out, dim=1)
    accuracy = compute_accuracy(preds, yb)
    #print(loss.item())

    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item(), accuracy
        
    return loss.item(), accuracy

def compute_accuracy(preds, labels):
    
    return np.mean(np.equal(preds.numpy(), labels.numpy()))



def run_epoch(model, train_dl, dev_dl, opt, loss_func):
    
    train_loss_batch = []
    train_accuracy_batch = []
    
    dev_loss_batch = []
    dev_accuracy_batch = []
    
    model.train()
    for xb, yb in train_dl:
        lossb, accuracyb = loss_batch(model, loss_func, xb, yb, opt)
        train_loss_batch.append(lossb)
        train_accuracy_batch.append(accuracyb)
    
    model. eval()
    with torch.no_grad():
        for xb, yb in dev_dl:
            lossb, accuracyb = loss_batch(model, loss_func, xb, yb)
            dev_loss_batch.append(lossb)
            dev_accuracy_batch.append(accuracyb)
            
    return np.mean(train_loss_batch), np.mean(train_accuracy_batch), np.mean(
        dev_loss_batch), np.mean(dev_accuracy_batch)
            
def fit(epochs, model, loss_func, opt, train_dl, validate_dl):
    train_loss_epoch = []
    train_acc_epoch = []
    dev_loss_epoch = []
    dev_acc_epoch = []
    
    for epoch in range(epochs):
        train_loss, train_acc, dev_loss, dev_acc = run_epoch(model, train_dl, dev_dl, opt, loss_func)
        print('epoch {}, train loss {}, train accuracy {}, dev accuracy {}'.format(
            epoch, train_loss, train_acc,dev_acc)
             )
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)
        dev_loss_epoch.append(dev_loss)
        dev_acc_epoch.append(dev_acc)
        
    
     
    plt.plot(np.arange(epochs)+1, train_loss_epoch, 'o-b')
    plt.plot(np.arange(epochs)+1, train_acc_epoch, 'o-r')
    plt.plot(np.arange(epochs)+1, dev_acc_epoch, 'o-g')
    plt.show()
    

    



train_dl, dev_dl = get_data(train_ds, dev_ds, batch_size = 64)

model, opt = get_model(mlp, 'adam', weight_decay = weight_decay)

fit(epochs, model, loss_func, opt, train_dl, dev_dl)



# MLP + DenseNet.
# DenseNet estimates the f* in a similar way as Taylor expansion. 
# for layer i, it uses concatenated inputs from all previous layers.

class _dense_layer(nn.Sequential):
    
    pass



# cutout

def get_model_old(model, opt_method, **kwargs):
    if opt_method.lower() == 'sgd':
        try:
            lr = kwargs['lr']
            momentum = kwargs['momentum']
        except KeyError as e:
            print(e, 'not defined')
            raise
        else:
            return model, torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum) 
        
    elif opt_method.lower() == 'adam':
        try:
            lr = kwargs['lr']
        except KeyError as e:
            print(e, 'not defined')
            raise
        else:
            return model, torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise ValueError('cannot find optimization method')
       def fit_old(epochs, model, loss_func, opt, train_dl, validate_dl):
    for epoch in range(epochs):
        
        model.train()
        for xb,  yb in train_dl:
            #print(xb)
            #sys.exit(0)
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in validate_dl]
            )
            
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)
        print(np.mean(batch_accuracies))
        
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# In[ ]:


# Refs:
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
# https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#from sklearn import svm
#import lightgbm as lgb
import time
import os
import math
import logging

from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys

    
        
# list of parameters   

all_features = True
num_of_fold = 5
lr = .01
momentum = 0.1
weight_decay = 0.001
epochs = 10
loss_func = F.cross_entropy
plot = False

# set up logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# Prepare for the train and dev datasets

def conv_features_factory(feature, number):
    '''func factory for generating functions that convert 0-1 features
        input:
            feature (str): name of the feature, 'Soil_Type' or 'Wilderness_Area' in this dataset
            number (int): number of features
        output:
            func : function for coverting features. shall be used in df.apply(func, axis = 1)
    '''
    def conv(data):
        for i in range(1, number+1, 1):
            dColumn = feature + str(i)
            if data[dColumn] == 1:
                return i
    return conv

## select all features (as in the train dataset) or reduced features, where 'Soil_Type' and 'Wilderness_Area'
## features are combined.
if all_features:
    
    train = pd.read_csv('/kaggle/input/learn-together/train.csv')
    
else:
    train = pd.read_csv('/kaggle/input/learn-together/train.csv')
    num_Soil_type = len(train.filter(regex = 'Soil_Type*').columns)
    num_Wildness_Area = len(train.filter(regex = 'Wilderness_Area*').columns)

    convSoil = conv_features_factory('Soil_Type', num_Soil_type)
    convWildness_Area = conv_features_factory('Wilderness_Area', num_Wildness_Area)

    train['SoilType'] = train.apply(convSoil, axis=1)
    train['WildernessArea'] = train.apply(convWildness_Area, axis=1)

    train = train.drop(train.filter(regex = 'Soil_Type*').columns, axis = 1)
    train = train.drop(train.filter(regex = 'Wilderness_Area*').columns, axis = 1)

# Preprocess of the data (rescale the data based on the estimated mean and variance across all the features)
# apply to train, dev, and test datasets


def get_data(train_ds, dev_ds, batch_size):
    return (
        DataLoader(train_ds, batch_size = batch_size, shuffle = True),
        DataLoader(dev_ds, batch_size = batch_size )
    )

def get_model(model, opt_method, **kwargs):
    if opt_method.lower() == 'sgd':
        try:
            opt = torch.optim.SGD(model.parameters(), **kwargs)
        except TypeError as e:
            print(e)
            raise
        else:
            return model, opt
        
    elif opt_method.lower() == 'adam':
        try:
            opt = torch.optim.Adam(model.parameters(), **kwargs)
        except TypeError as e:
            print(e)
            raise
        else:
            return model, opt
    else:
        raise ValueError('cannot find optimization method')

def fit(epochs, model, loss_func, opt, train_dl, dev_dl):
    train_loss_epoch = []
    train_acc_epoch = []
    dev_loss_epoch = []
    dev_acc_epoch = []
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, dev_loss, dev_acc = run_epoch(model, train_dl, dev_dl, opt, loss_func)
        # print('epoch {}, train loss {}, train accuracy {}, dev accuracy {}'.format(
        #     epoch, train_loss, train_acc,dev_acc)
        #      )
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)
        dev_loss_epoch.append(dev_loss)
        dev_acc_epoch.append(dev_acc)
        
    #plot = False
    
    logging.debug('based on the last 5 epochs, accuracy of the dev set {} with a std of {}'.format(
        np.mean(dev_acc_epoch[-5:]),np.std(dev_acc_epoch[-5:])))
    if plot: 
        plt.plot(np.arange(epochs)+1, train_loss_epoch, 'o-b', label = 'train loss')
        plt.plot(np.arange(epochs)+1, train_acc_epoch, 'o-r', label = 'train accuracy')
        plt.plot(np.arange(epochs)+1, dev_acc_epoch, 'o-g', label = 'dev accuracy')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylim(0.4,1)
        plt.show()
        
    
    return np.mean(dev_acc_epoch[-5:]), np.std(dev_acc_epoch[-5:])



def loss_batch(model, loss_func, xb, yb, opt = None):
    out = model(xb)
    loss = loss_func(out, yb)
    preds = torch.argmax(out, dim=1)
    accuracy = compute_accuracy(preds, yb)
    #print(loss.item())

    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item(), accuracy
        
    return loss.item(), accuracy

def compute_accuracy(preds, labels):
    
    return np.mean(np.equal(preds.numpy(), labels.numpy()))



def run_epoch(model, train_dl, dev_dl, opt, loss_func):
    
    train_loss_batch = []
    train_accuracy_batch = []
    
    dev_loss_batch = []
    dev_accuracy_batch = []
    
    model.train()
    for xb, yb in train_dl:
        lossb, accuracyb = loss_batch(model, loss_func, xb, yb, opt)
        train_loss_batch.append(lossb)
        train_accuracy_batch.append(accuracyb)
    
    model. eval()
    with torch.no_grad():
        for xb, yb in dev_dl:
            lossb, accuracyb = loss_batch(model, loss_func, xb, yb)
            dev_loss_batch.append(lossb)
            dev_accuracy_batch.append(accuracyb)
            
    return np.mean(train_loss_batch), np.mean(train_accuracy_batch), np.mean(
        dev_loss_batch), np.mean(dev_accuracy_batch)


mlp = nn.Sequential()
mlp.add_module('bn1', nn.BatchNorm1d(54))
mlp.add_module('linear1', nn.Linear(54,54))
mlp.add_module('ReLU1', nn.ReLU())
mlp.add_module('bn2', nn.BatchNorm1d(54))
mlp.add_module('linear2', nn.Linear(54,49))
mlp.add_module('ReLU2', nn.ReLU())

mlp.add_module('bn3', nn.BatchNorm1d(49))

mlp.add_module('linear3', nn.Linear(49,7))
mlp.add_module('ReLU4', nn.ReLU())
mlp.add_module('bn4', nn.BatchNorm1d(7))

mlp.add_module('linear4', nn.Linear(7,7))


def get_kfold_data(dataset, label_name, num_of_fold):
    
    '''generate kfold data 
    for each label l_i, the number of samples (randomly) selected for the dev dataset
    is proportional to total number of samples with label l_i
    '''
    
    
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError('get_kfold_data works only for pandas dataframe')
    
    if label_name not in dataset.columns:
        raise ValueError('{} not in the dataset'.format(label_name))
    ordered_dateset = dataset.sort_values(by = label_name)
    cover_type_dict = ordered_dateset['Cover_Type'].value_counts().to_dict()

    labels = set(ordered_dateset['Cover_Type'])

    l = []

    start = 0
    
    for i in labels:
        
        indeces = np.random.choice(np.arange(cover_type_dict[i]), size = math.floor(
            cover_type_dict[i]/num_of_fold), replace = None) + start
        #print(indeces)

        start += cover_type_dict[i]
        l.append(indeces)

        
    dev_indeces = np.concatenate(l)
    #print(dataset.shape[0])
    train_indeces = np.delete(np.arange(dataset.shape[0]), dev_indeces)
    X = ordered_dateset.drop(['Id', 'Cover_Type'], axis = 1)
    y = ordered_dateset['Cover_Type'] - 1 # Cover_Type: 0--6

    return X.iloc[train_indeces], y.iloc[train_indeces], X.iloc[dev_indeces], y.iloc[dev_indeces]
    
        
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def kfold(dataset, label_name, num_of_fold):
    '''cross validation'''
    X_train, y_train, X_dev, y_dev = get_kfold_data(train, 'Cover_Type', num_of_fold)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_trans = scaler.transform(X_train)  
    X_dev_trans = scaler.transform(X_dev) 

    train_ds = TensorDataset(torch.tensor(X_train_trans, dtype = torch.float), 
                         torch.tensor(y_train.values))
    dev_ds = TensorDataset(torch.tensor(X_dev_trans, dtype = torch.float), 
                            torch.tensor(y_dev.values))
    means = []
    for i in range(num_of_fold):
        print('fold {}'.format(i))
        train_dl, dev_dl = get_data(train_ds, dev_ds, batch_size = 80)
        for i in mlp.modules():
            if isinstance(i, nn.Linear):
                i.reset_parameters()
        model, opt = get_model(mlp, 'adam', weight_decay = weight_decay)
        mean, _ = fit(epochs, model, loss_func, opt, train_dl, dev_dl)
        means.append(mean)
    print(np.mean(means))
    
    
kfold(train, 'Cover_Type', num_of_fold)

#X_transformed = scaler.transform(X)



# helper function for separate train and dev datasets




# helper function for cross-validation



# helper function for MLP


# MLP model (use as a reference case for comparing with MLP+DenseNet)















            

    




# MLP + DenseNet.
# DenseNet estimates the f* in a similar way as Taylor expansion. 
# for layer i, it uses concatenated inputs from all previous layers.

class _denselayer(nn.Sequential):
    
    pass




# In[ ]:


dev_ratio = 0.2

def split(features, labels, dev_ratio):
    '''inputs:
                features: features pandas array
                labelss: labels for the feature pandas array
                dev_ratio: the proportion of data that will be used in the dev sets
                
        outputs:
                a tuple of (features_train, labels_train, features_dev, labels_dev)
    '''
    if not isinstance(features, pd.DataFrame):
        raise TypeError('features and labels should be pandas dataframes')
    
    if features.shape[0] != labels.shape[0]:
        raise ValueError('features and labels are not compatiable')
        
        
    if dev_ratio <= 0 or dev_ratio >= 1:
        raise ValueError('illegal dev_ratio input, should be in (0,1)')
    
    dev_indeces = np.random.choice(np.arange(labels.shape[0]), 
                           size = math.floor(labels.shape[0]*dev_ratio), replace = None)
    
    train_indeces = np.delete(np.arange(labels.shape[0]), dev_indeces)
    
    logging.debug('{:.0f}% of data will be used for dev.'.format(dev_ratio*100))
    
    return X.iloc[train_indeces], y.iloc[train_indeces], X.iloc[dev_indeces], y.iloc[dev_indeces]

X_train, y_train, X_dev, y_dev = split(X, y, .2)

X_train_trans = scaler.transform(X_train)  
X_dev_trans = scaler.transform(X_dev) 
train_ds = TensorDataset(torch.tensor(X_train_trans, dtype = torch.float), 
                         torch.tensor(y_train.values))
dev_ds = TensorDataset(torch.tensor(X_dev_trans, dtype = torch.float), 
                            torch.tensor(y_dev.values))

