# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Concatenate
from keras.models import Model
import pandas as pd
import gc
from sklearn.preprocessing import OneHotEncoder

import os
import keras.backend as K

Y_train = pd.read_csv('../input/train.csv')
files = os.listdir('../input/train/')
input_shape = (512,512,4)


Y_labels = np.zeros((int(len(files)/4.),28))

for c,i in enumerate(list(Y_train.Target.str.split(' '))):
    for j in i:
        Y_labels[c,int(j)] += 1

from sklearn.metrics import accuracy_score, f1_score     
        
def IoU(y_true, y_pred, offset = 0.001):
    thr = 0.1
    mx = 0
    
    rng = np.arange(0.1,0.9)
    
    for i in rng:
        predictions = y_pred
        predictions[predictions>=i] = 1
        predictions[predictions<i] = 0
        
        inter = np.logical_and(y_true, predictions)
        union = np.logical_or(y_true, predictions)
        
        inter = np.sum(inter)
        union = np.sum(union)
        
        out = np.round((inter + offset)/(union + offset),3)
        if out > mx:
            mx = out
            thr = i
    
    ret = 'Max IoU achieved is ' + str(mx) + ' at threshold ' + str(thr)
    print(ret)
    


def f1_sc(y_true, y_pred):
    
    thr = 0.1
    mx = 0
    
    rng = np.arange(0.1,0.9)
    
    for i in rng:
        predictions = y_pred
        predictions[predictions>=i] = 1
        predictions[predictions<i] = 0
        
        out = f1_score(y_true = y_true, y_pred = predictions, average = 'macro')
        if out > mx:
            mx = out
            thr = i
    ret = 'Max F1 Score achieved is ' + str(mx) + ' at threshold ' + str(thr)

    print(ret)

def acc_sc(y_true, y_pred):

    thr = 0.1
    mx = 0
    
    rng = np.arange(0.1,0.9)
    
    for i in rng:
        predictions = y_pred
        predictions[predictions>=i] = 1
        predictions[predictions<i] = 0
        
        out = accuracy_score(y_true = y_true, y_pred = predictions)
        if out > mx:
            mx = out
            thr = i
    ret = 'Max F1 Score achieved is ' + str(mx) + ' at threshold ' + str(thr)

    print(ret)


def EMR(y_true,y_pred):
    total = K.cast(K.cast(K.sum(K.abs(y_true-K.round(y_pred)),axis = 1),'bool'),'int32')
    total = K.sum(total)
    num_of_s = K.cast(K.shape(y_true)[0],'int32')
    
    return (num_of_s-total)/num_of_s

##################################### Class Weights ################################################

class_list = []

for item in Y_train.Target.values:
    cls = [i for i in item.split(' ')]
    class_list += cls

final = np.ndarray.astype(np.array(class_list),int)

class_weights = dict(zip(np.arange(0,28),len(training_set) / (28 * np.bincount(final))))


##################################### Class Weights ################################################

def PrModel(input_shape):

    X_input = Input(input_shape)

    # Layer 1
    X = Conv2D(8, (4, 4), name = 'conv0', activation = 'relu')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Dropout(rate = 0.4)(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    
    # Layer 2
    X = Conv2D(16, (4, 4), strides = 2, name = 'conv1', activation = 'relu', padding = 'same')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Dropout(rate = 0.3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    # Layer 3
    X = Conv2D(16, (4, 4), strides = 2, name = 'conv2', activation = 'relu', padding = 'same')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Dropout(rate = 0.3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
    # Layer 4
    X = Conv2D(64, (4, 4), strides = 2, name = 'conv3', activation = 'relu', padding = 'valid')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Dropout(rate = 0.3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)
    
    # Layer 5
    X = Conv2D(128, (2, 2), name = 'conv4', activation = 'relu', padding = 'valid')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Dropout(rate = 0.3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool4')(X)
    
    
    X = Flatten()(X)
    X = Dense(256, activation='relu', name='fc1')(X)
    X = Dropout(rate = 0.3)(X)
    
    X = Dense(28, activation='sigmoid', name='fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='ProteinRecognizer')
    
    
    return model
    
full = PrModel((512,512,4))
full.summary()

full.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [EMR])

np.random.seed(0)

valid_idx = np.random.randint(0,len(list(Y_train.Id)), 128)
train_idx = set(np.arange(0,len(list(Y_train.Id)))) - set(valid_idx)

training_set = [list(Y_train.Id)[x] for x in train_idx]
valid_set = [list(Y_train.Id)[x] for x in valid_idx]

training_labels = Y_labels[list(train_idx)]
valid_labels = Y_labels[list(valid_idx)]

import cv2, gc


for epoch in range(0,1):
    
    batch_size = 32
    data = np.zeros((batch_size,512,512,4))

    path = '../input/train/'
    
    print('Start batching')
    
    for batches in range(0,int(len(training_set)/batch_size)):
        fl = 0
        for j in training_set[batches*batch_size:(batches+1)*batch_size]:
            green = cv2.imread(path + j + '_green.png', cv2.IMREAD_GRAYSCALE)
            red = cv2.imread(path + j + '_red.png', cv2.IMREAD_GRAYSCALE)
            blue = cv2.imread(path + j + '_blue.png', cv2.IMREAD_GRAYSCALE)
            yellow = cv2.imread(path + j + '_yellow.png', cv2.IMREAD_GRAYSCALE)
            
            tmp = np.stack((green,red,blue,yellow), axis = -1)/255.
            
            data[fl,:,:,:] = tmp
            fl += 1
                
        print('Data ready for batch: ' + str(batches+1) + '/' + str(int(len(training_set)/batch_size)))
        
        Y = training_labels[(batches*batch_size):((batches+1)*batch_size)]
        
        full.fit(x = data, y = Y, epochs = 1, class_weight = class_weights) 
        
        if batches %50 == 0:
            tst_data = np.zeros((len(valid_set),512,512,4))
            tst_fl = 0
            for tst_j in valid_set:
                tst_green = cv2.imread(path + tst_j + '_green.png', cv2.IMREAD_GRAYSCALE)
                tst_red = cv2.imread(path + tst_j + '_red.png', cv2.IMREAD_GRAYSCALE)
                tst_blue = cv2.imread(path + tst_j + '_blue.png', cv2.IMREAD_GRAYSCALE)
                tst_yellow = cv2.imread(path + tst_j + '_yellow.png', cv2.IMREAD_GRAYSCALE)
                
                tst_tmp = np.stack((tst_green,tst_red,tst_blue,tst_yellow), axis = -1)
                
                tst_data[tst_fl,:,:,:] = tst_tmp
                tst_fl += 1
            preds = full.predict(x = tst_data)
			
            print()
            print('-----------------------------------------------')
            acc_sc(y_true = valid_labels, y_pred = preds)
            IoU(y_true = valid_labels, y_pred = preds)
            f1_sc(y_true = valid_labels, y_pred = preds)
            print('-----------------------------------------------')
            print()
            
            

