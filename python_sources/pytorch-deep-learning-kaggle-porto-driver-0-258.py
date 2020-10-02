#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Bootcamp November 2017, GPU Computing for Data Scientists
# 
# #### Shlomo Kashani 
# 
# 
# ## 69-PyTorch-Kaggle-porto-driver
# 
# Web: https://www.meetup.com/Tel-Aviv-Deep-Learning-Bootcamp/events/241762893/
# 
# Notebooks: <a href="https://github.com/QuantScientist/Data-Science-PyCUDA-GPU"> On GitHub</a>
# 
# *Shlomo Kashani*
# 
# 
# ### Data
# - Download from Kaggle
# 
# ### Epochs
# I set epochs=500 because in Kaggle this runs on a CPU, change to 5000 for better results on a GPU

# # PyTorch Imports
# 

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import torch
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)

import numpy
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

import pandas
import pandas as pd

import logging
handler=logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
get_ipython().run_line_magic('matplotlib', 'inline')

# !pip install psutil
import psutil
import os
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

cpuStats()


# #  Global params

# In[ ]:


# fix seed
seed=17*19
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)    
# ! dir    


# 
# #  View the Data

# In[ ]:


import gc; gc.enable()
# !pip install xgboost
import xgboost as xgb
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost
import pandas as pd
import numpy as np
from sklearn import *
import sklearn

# Data params
TARGET_VAR= 'target'
BASE_FOLDER = '../input/'

# Read in our input data
df_train = pd.read_csv(BASE_FOLDER + '/train.csv')
df_test = pd.read_csv(BASE_FOLDER + '/test.csv')
# This prints out (rows, columns) in each dataframe
print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)

print('Columns:', df_train.columns)

y_train = df_train['target'].values
id_train = df_train['id'].values
id_test = df_test['id'].values
df_train.head()


# #  Train / Validation / Test Split

# In[ ]:


x_train = df_train.drop(['target', 'id'], axis=1)
x_test = df_test.drop(['id'], axis=1)

# Take a random 20% of the dataset as validation data
trainX, valX, trainY, valY = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
print('Train samples: {} Validation samples: {}'.format(len(trainX), len(valX)))

N_FEATURES=trainX.shape[1]


# #  From Numpy to PyTorch GPU tensors

# In[ ]:


use_cuda = torch.cuda.is_available()
# use_cuda = False


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)        
    print(x_data_np.shape)
    print(type(x_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")    
        X_tensor = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    
    else:
        lgr.info ("Using the CPU")
        X_tensor = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch
    
    print(type(X_tensor.data)) # should be 'torch.cuda.FloatTensor'            
    print((X_tensor.data.shape)) # torch.Size([108405, 29])
    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np):    
    y_data_np=y_data_np.reshape((y_data_np.shape[0],1)) # Must be reshaped for PyTorch!
    print(y_data_np.shape)
    print(type(y_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")            
    #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float        
    else:
        lgr.info ("Using the CPU")        
    #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #         
        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        

    print(type(Y_tensor.data)) # should be 'torch.cuda.FloatTensor'
    print(y_data_np.shape)
    print(type(y_data_np))    
    return Y_tensor


# In[ ]:


# class ConvRes(nn.Module):
#     def __init__(self, insize, outsize):
#         super(ConvRes, self).__init__()
#         drate = .3
#         self.math = nn.Sequential(
#             nn.BatchNorm1d(insize),            
#             torch.nn.Conv1d(insize, outsize, kernel_size=2, padding=2),
#             nn.PReLU(),
#         )
#     def forward(self, x):
#         return self.math(x)

# class ConvCNN(nn.Module):
#     def __init__(self, insize, outsize, kernel_size=7, padding=2, pool=2, avg=True):
#         super(ConvCNN, self).__init__()
#         self.avg = avg
#         self.math = torch.nn.Sequential(
#             torch.nn.Conv1d(insize, outsize, kernel_size=kernel_size, padding=padding),
#             torch.nn.BatchNorm1d(outsize),
#             torch.nn.LeakyReLU(),
#             torch.nn.MaxPool1d(pool),
#         )        
#     def forward(self, x):
#         x = self.math(x)        
#         return x

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()        

#         self.cnn1 = ConvCNN(N_FEATURES, 64, kernel_size=7, pool=4, avg=False)
#         self.cnn2 = ConvCNN(64, 64, kernel_size=5, pool=2, avg=True)
#         self.cnn3 = ConvCNN(64, 32, kernel_size=5, pool=2, avg=True)
#         self.res1 = ConvRes(32, 64)

#         self.features = nn.Sequential(
#             self.cnn1,
# #             self.cnn2,
# #             self.cnn3,
# #             self.res1,
#         )

#         self.classifier = torch.nn.Sequential(
#             nn.Linear(1024, 1),
#         )
#         self.sig = nn.Sigmoid()

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         x = self.sig(x)
#         return x
    
    
X_tensor_train= XnumpyToTensor(trainX) # default order is NBC for a 3d tensor, but we have a 2d tensor
X_shape=X_tensor_train.data.size()


n_mult_factor=9
n_input= trainX.shape[1]
n_hidden= n_input * n_mult_factor
n_output=1
n_input_rows=trainX.shape[0]
n_cnn_kernel=7
n_padding=4
n_max_pool1d=2

DEBUG_ON=True
def debug(msg, x):
    if DEBUG_ON:
        print (msg + ', (size():' + str (x.size()))
    
class CNNNumerAI(nn.Module):    
    def __init__(self, n_input, n_hidden, n_output,n_cnn_kernel, n_mult_factor, n_padding,n_max_pool1d):
        super(CNNNumerAI, self).__init__()    
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output= n_output 
        self.n_cnn_kernel=n_cnn_kernel
        self.n_mult_factor=n_mult_factor
        self.n_padding=n_padding
        self.n_max_pool1d=n_max_pool1d
        self.n_l1=int((n_mult_factor * self.n_input) * (n_padding + 1) / n_max_pool1d)
                    
        self.features = nn.Sequential(  
            torch.nn.Conv1d(self.n_input, self.n_hidden,kernel_size=(self.n_cnn_kernel,), stride=(1,), padding=(self.n_padding,)),                                             
            torch.nn.LeakyReLU(),            
            torch.nn.MaxPool1d(kernel_size=self.n_max_pool1d),
                                    
        )                        
                
        linear4=torch.nn.Linear(int(self.n_hidden), 1)
        torch.nn.init.xavier_uniform(linear4.weight)        
        
        self.classifier = torch.nn.Sequential
                                 (
                                    linear4
                                  )                                 
        self.sig=nn.Sigmoid()
                
        
    def forward(self, x):
        varSize=x.data.shape[0] # must be calculated here in forward() since its is a dynamic size                          
        # for CNN  
        x=x.contiguous() 
        x = x.view(varSize,self.n_input,1)
        debug('after view',x)   
        x=self.features(x)
        debug('after CNN',x)           
        x = x.view(varSize,int(self.n_hidden)) 
        debug('after 2nd view',x)                  
        x=self.classifier(x)   
        debug('after self.out',x)   
        x=self.sig(x)
        return x

net = CNNNumerAI(n_input, n_hidden, n_output,n_cnn_kernel, n_mult_factor, n_padding, n_max_pool1d)    
print(net)

if use_cuda:
    net=net.cuda() 
b = net(X_tensor_train)

print ('(b.size():' + str (b.size()))    

LR = 0.005

optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=5e-5) #  L2 regularization
loss_func=torch.nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

if use_cuda:
    lgr.info ("Using the GPU")    
    net.cuda()
    loss_func.cuda()
#     cudnn.benchmark = True

lgr.info (optimizer)
lgr.info (loss_func)


# # Training set

# In[ ]:


from __future__ import division

import time
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

# for windows
torch.backends.cudnn.enabled=False

start_time = time.time()    
epochs=1500 # change to 5000 for better results
div_factor=100
all_losses = []
loss_arr =[]
DEBUG_ON=False

print (net)

X_tensor_train= XnumpyToTensor(trainX)
Y_tensor_train= YnumpyToTensor(trainY)
print(type(X_tensor_train.data), type(Y_tensor_train.data)) # should be 'torch.cuda.FloatTensor'

# CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
# X_tensor_train=X_tensor_train.contiguous()
# Y_tensor_train=Y_tensor_train.contiguous()
                
# From here onwards, we must only use PyTorch Tensors
for step in range(epochs):    
    out = net(X_tensor_train)                 # input x and predict based on x
    cost = loss_func(out, Y_tensor_train)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    cost.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
                   
        
    if step % div_factor == 0:        
        loss = cost.data[0]
        all_losses.append(loss)
        print(step, cost.data.cpu().numpy())
        # RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays). 
        # Use .cpu() to move the tensor to host memory first.        
        prediction = (net(X_tensor_train).data).float() # probabilities         
#         prediction = (net(X_tensor).data > 0.5).float() # zero or one
#         print ("Pred:" + str (prediction)) # Pred:Variable containing: 0 or 1
#         pred_y = prediction.data.numpy().squeeze()            
        pred_y = prediction.cpu().numpy().squeeze()
        target_y = Y_tensor_train.cpu().data.numpy()
                        
        tu = (log_loss(target_y, pred_y),roc_auc_score(target_y,pred_y ), 2*roc_auc_score(target_y,pred_y ) - 1)
        print ('LOG_LOSS={}, ROC_AUC={}, GINI={}'.format(*tu))  
        
        loss_arr.append(cost.cpu().data.numpy()[0])
                
end_time = time.time()
print ('{} {:6.3f} seconds'.format('GPU:', end_time-start_time))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(all_losses)
plt.show()

false_positive_rate, true_positive_rate, thresholds = roc_curve(target_y,pred_y)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('GINI:' + str(2*roc_auc_score(target_y,pred_y ) - 1))
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


net.eval()
# Validation data
print (valX.shape)
print (valY.shape)

X_tensor_val= XnumpyToTensor(valX)
Y_tensor_val= YnumpyToTensor(valY)


print(type(X_tensor_val.data), type(Y_tensor_val.data)) # should be 'torch.cuda.FloatTensor'

predicted_val = (net(X_tensor_val).data).float() # probabilities 
# predicted_val = (net(X_tensor_val).data > 0.5).float() # zero or one
pred_y = predicted_val.cpu().numpy()
target_y = Y_tensor_val.cpu().data.numpy()                

print (type(pred_y))
print (type(target_y))


print ('\n')
tu = (log_loss(target_y, pred_y),roc_auc_score(target_y,pred_y ), 2*roc_auc_score(target_y,pred_y ) - 1)
print ('LOG_LOSS={}, ROC_AUC={}, GINI={}'.format(*tu))  
        
false_positive_rate, true_positive_rate, thresholds = roc_curve(target_y,pred_y)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('GINI=' + str(2*roc_auc_score(target_y,pred_y ) - 1))
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# print (pred_y)


# In[ ]:


# Submission


# In[ ]:


# X_df_test = pd.read_csv(BASE_FOLDER + '/test.csv')
# print('Test shape:', X_df_test.shape)
# print('Columns:', X_df_test.columns)
# id_test = X_df_test['id'].values
# X_df_test=X_df_test.apply(lambda x: pandas.to_numeric(x, errors='ignore'))


# print (X_df_test.shape)
# columns = ['id', 'target']
# df_pred=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)


# for index, row in X_df_test.iterrows():
#     rwo_no_id=row.drop('id')    
# #     print (rwo_no_id.values)    
#     x_data_np = np.array(rwo_no_id.values, dtype=np.float32)        
#     if use_cuda:
#         X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    
#     else:
#         X_tensor_test = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch
                    
#     X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors            
#     predicted_val = (net(X_tensor_test).data).float() # probabilities     
#     p_test =   predicted_val.cpu().numpy().item() # otherwise we get an array, we need a single float
    
#     df_pred = df_pred.append({'id':row['id'], 'target':p_test},ignore_index=True)
# #     df_pred = df_pred.append({'id':row['id'].astype(int), 'probability':p_test},ignore_index=True)

# df_pred.head(5)


# In[ ]:


# df_pred.id=df_pred.id.astype(int)

# def savePred(df_pred, loss):
# #     csv_path = 'pred/p_{}_{}_{}.csv'.format(loss, name, (str(time.time())))
#     csv_path = 'pred/pred_{}_{}.csv'.format(loss, (str(time.time())))
#     df_pred.to_csv(csv_path, columns=('id', 'target'), index=None)
#     print (csv_path)
    
# savePred (df_pred, str(2*roc_auc_score(target_y,pred_y ) - 1))


# In[ ]:





# In[ ]:





# In[ ]:




