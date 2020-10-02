#!/usr/bin/env python
# coding: utf-8

# **Acknowledgements **
# 
# This kernel was inspired by 
# 1. The deep learning book by Michael Nielsen particularly because amazing storytelling of some confusing ideas and concepts of Neural Networks. http://neuralnetworksanddeeplearning.com/chap6.html
# 2. Andrew ng Deep learning course (Course 1,2,4) on Coursera
# 3. Kaggle kernel https://www.kaggle.com/gomezp/complete-beginner-s-guide-eda-keras-lb-0-93   

# **Introduction**
# 
# As a noob, This kernel is not develop some state of the art Neural net and immediately be at the frontiers. The main purpose of this kernel is to focus on fundamentals and that's it. 
# Following are the steps I think is right for the gradual understanding of the model:
# 
# 1. Creating a workable CNN using keras
# 2. Improving the CNN with tweaking hyperparameters and new architecture (with varying depth) and understand their effect on the network's performance.
# 
# 3. Introducing Resnets/ Inception net/ and other models and use transfer learning to train our models.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from glob import glob
import gc
import sys
from tqdm import tqdm_notebook, trange
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Step 1. loading the data in the proper format 
# * X: input images rgb representation
# * y: labels for the images 

# In[ ]:


path = "../input/histopathologic-cancer-detection/"
train_path = path + 'train/'
test_path = path + 'test/'

df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))})


df['id'] = df.path.map(lambda x: x.split('/')[4].split(".")[0]) # keep only the file names in 'id'
labels = pd.read_csv(path+"train_labels.csv")                   # read the provided labels
df = df.merge(labels, on = "id")                                # merge labels and filepaths                                           
df.head(3)


# In[ ]:


import tensorflow as tf

from keras.models import load_model

new_model = tf.keras.models.load_model('../input/128batch/my_model4.h5')


# To get the rgb numeric representation, we will use the cv2 library by opencv for computer vision .cv2.imread takes the input images path and outputs the images in rgb format

# In[ ]:


#loading the data of N inputs and labels
import  cv2                   
def load_data(df,N):
    X= np.zeros([N,96,96,3],dtype= np.uint8)
    y = np.zeros([N,1], dtype = np.uint8)
    for i, row in tqdm_notebook(df.iterrows(), total = N):
        if i == N:
            break
        X[i] = cv2.imread(row['path'])
        y[i] = row['label']
        
    
    return X,y

X,y = load_data(df= df, N = len(df))
    


# In[ ]:


df['label'].mean()


# Due to limited availability of the ram(13 GB), we dont want to create any new variable ( X_train, X_val, Y_train, Y_test). as that will overflow the RAM. 
# 1. Instead, we shuffle the data ,so that the distribution of the data is not skewed. We dont want some particular types of image just in training data. That will lead to poor generalization by the network for the test and validation set.
# 2. We replace (to deal with limited RAM) the X,y with the new shuffled indexes

# In[ ]:


training_portion = 0.8 # Specify training/validation ratio
split_idx = int(np.round(training_portion * y.shape[0])) #Compute split idx

np.random.seed(42) #set the seed to ensure reproducibility

#shuffle
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]


# In[ ]:


print(X.shape[0])
print(y.shape[0])


# In[ ]:


# import tensorflow as tf
# import keras
# from keras import optimizers
# import tensorflow.keras.models
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
# from tensorflow.keras.models import Sequential

# dropout_conv = 0.3
# dropout_dense = 0.5
# #creating the new_model 

# new_model = Sequential()
# new_model.add(Conv2D(filters = 32,kernel_size = 3, activation = 'relu', padding = 'same',input_shape= [96,96,3]))
# new_model.add(MaxPooling2D(pool_size = 2))
# new_model.add(Dropout(dropout_conv))

# new_model.add(Conv2D(filters = 64,kernel_size = 3, activation = 'relu', padding = 'same'))
# new_model.add(MaxPooling2D(pool_size = 2))
# new_model.add(Dropout(dropout_conv))

# new_model.add(Conv2D(filters = 128,kernel_size = 3, activation = 'relu', padding = 'same'))
# new_model.add(MaxPooling2D(pool_size = (2,2)))
# new_model.add(Dropout(dropout_conv)) 
    
# new_model.add(Flatten())
# new_model.add(Dense(256, activation ='relu'))
# new_model.add(Dropout(dropout_dense))

# new_model.add(Dense(1,activation = 'sigmoid'))

# new_model.compile(loss = 'binary_crossentropy',
#              optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
#              metrics = ['accuracy'])

# # new_model.fit(X_train, Y_train, batch_size = 100, epochs = 15, validation_data = (X_test, Y_test))
# new_model.fit(X[:split_idx], y[:split_idx], epochs=10, batch_size= 32,validation_data = (X[split_idx:], y[split_idx:]))


# Training on the 80% of the dataset and validation on the rest. The accuracy and loss calculated are for each mini-batch and not for the whole dataset. 
# 
# 
# Later, after initial run, we can get the correct accuracy and loss for the whole dataset,Now, we are averaging the accuracy for each mini-batch to calculate for the whole epoch. (A rough estimation)

# In[ ]:


epochs = 7
batch_size = 128

for epoch in range(epochs):
    loss, accuracy  = 0,0
    
    iterations = int(split_idx/batch_size)
    
#     with trange(0,split_idx, batch_size) as t:
#         for i in t:
#             start_ind = i
#             x_batch = X[i:i + batch_size]
#             y_batch = y[i:i + batch_size]
            
            

    with trange(iterations) as t:
        for i in t:
            start_ind = i * batch_size
            x_batch = X[start_ind: start_ind + batch_size]
            y_batch = y[start_ind: start_ind + batch_size]

            metrics = new_model.train_on_batch(x_batch, y_batch)

            loss = loss + metrics[0]   # calculating loss and accuracy for that mini- batch
            accuracy = accuracy +metrics[1]

            t.set_description('running_training_epoch '+ str(epoch))
            t.set_postfix(loss = "%.2f" % round(loss/(i+1),2), accuracy = "%.2f" % round(accuracy/(i+1),2) )


# After training on the training dataset, we trained the weights and biases of the network. Now, we input the validation images into the network and get the output and calculate the cost and accuracy for the validation set
# 

# In[ ]:


X = X[split_idx:]
y = y[split_idx:]

iterations = int((y.shape[0])/batch_size)
# iterations = 20
loss, accuracy = 0,0 
with trange(iterations) as t:
    for i in t:

        start_idx = i * batch_size #starting index of the current batch
        x_batch = X[start_idx :start_idx + batch_size] #the current batch
        y_batch = y[start_idx:start_idx + batch_size] #the labels for the current batch


        metrics = new_model.test_on_batch(x_batch, y_batch)

        loss = loss + metrics[0]   # calculating loss and accuracy for that mini- batch
        accuracy = accuracy +metrics[1]

        t.set_description('running_validation ')
        t.set_postfix(loss = round(loss/(i+1),2), accuracy = round(accuracy/(i+1),2) )


# In[ ]:


X = None
y = None
df= None
gc.collect();


# After training and validation, we input the test image into the network and take the output and upload in kaggle to get the testing accuracy

# In[ ]:


test_df = pd.DataFrame({'path': glob(os.path.join(test_path,'*.tif'))})
test_df['id'] = test_df['path'].map(lambda x : x.split('/')[4].split('.')[0])

test_df['image'] = test_df['path'].map(cv2.imread)
X = np.stack(test_df['image'], axis = 0)

print(X.shape)


# In[ ]:


predictions = new_model.predict(X, verbose = 1)
test_df['label'] = predictions


# In[ ]:


df = None
gc.collect()
submission = pd.DataFrame()
submission['id'] = test_df['id']

submission['label'] = predictions

submission.head()


# In[ ]:


submission.to_csv("submission.csv", header = True, index = False)


# In[ ]:


from keras.models import load_model

new_model.save('my_model4.h5')


# In[ ]:




