
# coding: utf-8

# In[1]:

## Model pictures and acuuracy results : https://github.com/shubhanshu786/learning/tree/master/obfuscated-multiclassification
## Validation set accuracy : ~85%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Dropout, Softmax
from keras.callbacks import TensorBoard
from time import time


# In[2]:


#data can be downloaded from https://www.kaggle.com/alaeddineayadi/obfuscated-multiclassification
with open('../input/xtrain_obfuscated.txt') as fp:
    data = fp.read().split('\n')
with open('../input/ytrain.txt') as fp:
    label = fp.read().split('\n')

if data[-1] == '':
    data = data[:-1]
    
if label[-1] == '':
    label = label[:-1]


# In[3]:


max_char = 26
max_len = 0
for sent in data:
    max_len = max(max_len, len(sent))


# In[4]:


#One hot encoding for labels
for i in range(len(label)):
    label[i] = int(label[i])
label = np.asarray(label)
ohe = OneHotEncoder(sparse=False)
label = label.reshape(-1,1)
y_train_data = ohe.fit_transform(label)


# In[5]:


# One hot feature generation
# MaxLen = 452. 
# Other sentences are 0 padded
X_train = []
for i in range(len(data)):
    temp = np.zeros((max_len, max_char))
    for j in range(len(data[i])):
        temp[j][ord(data[i][j])-97] = 1
    X_train.append(temp)
X_train = np.asarray(X_train)
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1], X_train.shape[2],1))


# In[14]:


# 1) CNN in text data can be used to capture local/temporal dependencies in data.
# eg, in this case i am performing 8 different CNN operations over same data, with different convolution size
# 2) Here every convolution size means, CNN is trying to capture useful pattern in data in that size frame.
# eg, with convolution size 5, CNN will check for all 5-grams(chars in this case) in sentence and try to learn 
# useful pattern in data
# 3) In case of text data kernal_size = pattern_size X feature_size. Since we need to consider whole char a 
# the same time, that why feature_size is important factor here

convSize_1 = 3
convSize_2 = 4
convSize_3 = 5
convSize_4 = 6
convSize_5 = 7
convSize_6 = 8
convSize_7 = 9
convSize_8 = 10

# Convolution with different kernal sizes
inputLayer = Input(shape=(X_train[0].shape[0],X_train[0].shape[1],1))
convLayer_1 = Conv2D(filters=128, kernel_size=(convSize_1,X_train[0].shape[1]))(inputLayer)
convLayer_2 = Conv2D(filters=128, kernel_size=(convSize_2,X_train[0].shape[1]))(inputLayer)
convLayer_3 = Conv2D(filters=128, kernel_size=(convSize_3,X_train[0].shape[1]))(inputLayer)
convLayer_4 = Conv2D(filters=128, kernel_size=(convSize_4,X_train[0].shape[1]))(inputLayer)
convLayer_5 = Conv2D(filters=128, kernel_size=(convSize_5,X_train[0].shape[1]))(inputLayer)
convLayer_6 = Conv2D(filters=128, kernel_size=(convSize_6,X_train[0].shape[1]))(inputLayer)
convLayer_7 = Conv2D(filters=128, kernel_size=(convSize_7,X_train[0].shape[1]))(inputLayer)
convLayer_8 = Conv2D(filters=128, kernel_size=(convSize_8,X_train[0].shape[1]))(inputLayer)

#Dropout to prevent overfitting
dropout_1 = Dropout(0.5)(convLayer_1)
dropout_2 = Dropout(0.5)(convLayer_2)
dropout_3 = Dropout(0.5)(convLayer_3)
dropout_4 = Dropout(0.5)(convLayer_4)
dropout_5 = Dropout(0.5)(convLayer_5)
dropout_6 = Dropout(0.5)(convLayer_6)
dropout_7 = Dropout(0.5)(convLayer_7)
dropout_8 = Dropout(0.5)(convLayer_8)

#Maxpool within the features
maxPool_1 = MaxPooling2D(pool_size=(max_len-convSize_1+1, 1))(dropout_1)
maxPool_2 = MaxPooling2D(pool_size=(max_len-convSize_2+1, 1))(dropout_2)
maxPool_3 = MaxPooling2D(pool_size=(max_len-convSize_3+1, 1))(dropout_3)
maxPool_4 = MaxPooling2D(pool_size=(max_len-convSize_4+1, 1))(dropout_4)
maxPool_5 = MaxPooling2D(pool_size=(max_len-convSize_5+1, 1))(dropout_5)
maxPool_6 = MaxPooling2D(pool_size=(max_len-convSize_6+1, 1))(dropout_6)
maxPool_7 = MaxPooling2D(pool_size=(max_len-convSize_7+1, 1))(dropout_7)
maxPool_8 = MaxPooling2D(pool_size=(max_len-convSize_8+1, 1))(dropout_8)

#Flatten all the data from CNN model
flatten_1 = Flatten()(maxPool_1)
flatten_2 = Flatten()(maxPool_2)
flatten_3 = Flatten()(maxPool_3)
flatten_4 = Flatten()(maxPool_4)
flatten_5 = Flatten()(maxPool_5)
flatten_6 = Flatten()(maxPool_6)
flatten_7 = Flatten()(maxPool_7)
flatten_8 = Flatten()(maxPool_8)

#Merge all the 8 layers data
mergedLayer = Concatenate(axis=1)([flatten_1, flatten_2, flatten_3, flatten_4, flatten_5, flatten_6, flatten_7, flatten_8])

#Dense layers and so on
dense_1 = Dense(1024, activation='relu')(mergedLayer)
dropout_9 = Dropout(0.5)(dense_1)
dense_2 = Dense(128, activation='relu')(dropout_9)
dropout_10 = Dropout(0.5)(dense_2)

result = Dense(12, activation='softmax')(dropout_10)


# In[15]:


model = Model(inputLayer, result)
model.summary()


# In[16]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir='./logs/{}'.format(time()))


# In[17]:

#For commiting just putting epochs=1. Usually put 50 for stable results

model.fit(X_train, y_train_data, batch_size=128, verbose=1, epochs=1, shuffle=True, validation_split=0.2)


# In[11]:

## Accuracy achieved at validation set : ~85%
## check at https://github.com/shubhanshu786/learning/tree/master/obfuscated-multiclassification


model.save('8_layer_conv.h5')
