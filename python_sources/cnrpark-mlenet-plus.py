#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
from IPython.display import display
from PIL import Image
import cv2
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.utils import np_utils
import math
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dense, Activation, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


# In[ ]:


source_path = "../input/"
print(os.listdir(source_path))
source_label = source_path + "splits-path/splits/splits/"
ROW = 150
COL = 150
CHANNEL = 3
num_classes = 2


# In[ ]:


def reshaped_image(image):
    return transform.resize(image,(ROW, COL, CHANNEL))


# In[ ]:


def loadImageInRange(ImgPath):
    Batch_Images = []
    for path in ImgPath:
        img = cv2.imread(path)
        Batch_Images.append(reshaped_image(img))
    return np.array(Batch_Images)


# In[ ]:


def loadDataPath(label_path, data_path):
    fp = os.path.join(source_label, label_path)
    _PathImgs = []
    _Labels = []
    with open(fp) as f:
        contents = f.readlines()
        contents = [x.strip() for x in contents]
        for content in contents:
            content = content.split()
            path = os.path.join(data_path, content[0])
            _PathImgs.append(path)
            _Labels.append(int(content[1]))
    return np.array(_PathImgs), np.array(_Labels)


# In[ ]:


label_path = "CNRParkAB/all.txt"
data_path = source_path + "cnrpark-patches/cnrpark-patches-150x150/CNRPark-Patches-150x150/"
Train_PathDatas, Train_Labels = loadDataPath(label_path, data_path)
Train_Labels = np_utils.to_categorical(Train_Labels, num_classes)
Train_Labels.shape


# In[ ]:


label_path = "CNRPark-EXT/camera1.txt"
data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
Train_PathDatas_c1, Train_Labels_c1 = loadDataPath(label_path, data_path)
Train_Labels_c1 = np_utils.to_categorical(Train_Labels_c1, num_classes)
label_path = "CNRPark-EXT/camera2.txt"
data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
Train_PathDatas_c2, Train_Labels_c2 = loadDataPath(label_path, data_path)
Train_Labels_c2 = np_utils.to_categorical(Train_Labels_c2, num_classes)
label_path = "CNRPark-EXT/camera3.txt"
data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
Train_PathDatas_c3, Train_Labels_c3 = loadDataPath(label_path, data_path)
Train_Labels_c3 = np_utils.to_categorical(Train_Labels_c3, num_classes)


# In[ ]:


Train_PathDatas = np.concatenate((Train_PathDatas, Train_PathDatas_c1, Train_PathDatas_c2, Train_PathDatas_c3),axis=0 )
Train_Labels = np.concatenate((Train_Labels, Train_Labels_c1, Train_Labels_c2, Train_Labels_c3),axis=0)
(Train_PathDatas.shape, Train_Labels.shape)


# In[ ]:


def newGenerator(batch_size, PathDatas, LabelsOfData, num_batchs):
    while 1:
        for i in range(num_batchs):
            if i == num_batchs - 1:
                Imgs_Data = loadImageInRange(PathDatas[i*batch_size:])
                Labels_Data = LabelsOfData[i*batch_size:]
                yield Imgs_Data, Labels_Data
            else:
                Imgs_Data = loadImageInRange(PathDatas[i*batch_size:(i+1)*batch_size])
                Labels_Data = LabelsOfData[i*batch_size:(i+1)*batch_size]
                yield Imgs_Data, Labels_Data


# In[ ]:


def model_mLeNet(input_shape):
    
    mLeNet = Sequential()
    # Layer 1
    mLeNet.add(Conv2D(30, (11,11), input_shape = input_shape, strides = (4,4),  padding='same'))
    mLeNet.add(Activation('relu'))
    mLeNet.add(MaxPooling2D(pool_size=(5, 5), strides=(5,5)))
    
    # Layer 2
    mLeNet.add(Conv2D(20, (5,5), strides = (1,1),  padding='same'))
    mLeNet.add(Activation('relu'))
    mLeNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    # Layer 3
    mLeNet.add(Flatten())
    mLeNet.add(Dense(100, activation = 'relu'))
    
    # Layer 4
    mLeNet.add(Dense(num_classes, activation = 'softmax'))
    
    sgd = optimizers.SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
    
    mLeNet.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(mLeNet.summary())
    return mLeNet


# In[ ]:


input_shape = (ROW, COL, CHANNEL)
#model = model_mLeNet(input_shape)
from keras.models import load_model
model = load_model('../input/cnrparkmodel/mLeNet_keras.h5')


# In[ ]:


batch_size = 64
epochs = 5
num_samples = Train_Labels.shape[0]
num_batchs = math.ceil(num_samples / batch_size)


# In[ ]:


hist = model.fit_generator(newGenerator(batch_size=batch_size, 
                                        PathDatas=Train_PathDatas, 
                                        LabelsOfData=Train_Labels, 
                                        num_batchs=num_batchs),
                           steps_per_epoch=num_batchs, 
                           epochs=epochs, 
                           verbose=1
                          )


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

train_loss=hist.history['loss']
train_acc=hist.history['acc']
epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'r', label='train_loss')
plt.title('train_loss')
plt.legend()
plt.figure()

plt.plot(epochs,train_acc,'r', label='train_acc')
plt.title('train_acc')
plt.legend()
plt.figure()


# In[ ]:


model.save_weights('mLeNet_weights.h5')
model.save('mLeNet_keras.h5')


# In[ ]:


#from keras.models import load_model
#model = load_model('../input/mlenetmodel/mLeNet_keras.h5')


# In[ ]:



#label_path = "CNRPark-EXT/camera4.txt"
#data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
#Test_PathDatas_c4, Test_Labels_c4 = loadDataPath(label_path, data_path)
#Test_Labels_c4 = np_utils.to_categorical(Test_Labels_c4, num_classes)

#label_path = "CNRPark-EXT/camera5.txt"
#data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
##Test_PathDatas_c5, Test_Labels_c5 = loadDataPath(label_path, data_path)
#Test_Labels_c5 = np_utils.to_categorical(Test_Labels_c5, num_classes)

#label_path = "CNRPark-EXT/camera6.txt"
#data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
#Test_PathDatas_c6, Test_Labels_c6 = loadDataPath(label_path, data_path)
#Test_Labels_c6 = np_utils.to_categorical(Test_Labels_c6, num_classes)

#label_path = "CNRPark-EXT/camera7.txt"
#data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
#Test_PathDatas_c7, Test_Labels_c7 = loadDataPath(label_path, data_path)
#Test_Labels_c7 = np_utils.to_categorical(Test_Labels_c7, num_classes)

#label_path = "CNRPark-EXT/camera8.txt"
#data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
#Test_PathDatas_c8, Test_Labels_c8 = loadDataPath(label_path, data_path)
#Test_Labels_c8 = np_utils.to_categorical(Test_Labels_c8, num_classes)

#label_path = "CNRPark-EXT/camera9.txt"
#data_path = source_path + "cnrpark-ext/cnr-ext-patches-150x150/PATCHES/"
#Test_PathDatas_c9, Test_Labels_c9 = loadDataPath(label_path, data_path)
#Test_Labels_c9 = np_utils.to_categorical(Test_Labels_c9, num_classes)

#Test_PathDatas = np.concatenate((Test_PathDatas_c4, Test_PathDatas_c5, Test_PathDatas_c6, Test_PathDatas_c7, Test_PathDatas_c8, Test_PathDatas_c9),axis=0 )
#Test_Labels = np.concatenate((Test_Labels_c4, Test_Labels_c5, Test_Labels_c6, Test_Labels_c7, Test_Labels_c8, Test_Labels_c9),axis=0)
#(Test_PathDatas.shape, Test_Labels.shape)


# In[ ]:


#batch_size = 64
#num_tests = Test_Labels.shape[0]
#num_batchs_test = math.ceil(num_tests / batch_size)


# In[ ]:



#score = model.evaluate_generator(newGenerator(batch_size=batch_size, 
#                                    PathDatas=Test_PathDatas, 
#                                    LabelsOfData=Test_Labels, 
#                                    num_batchs=num_batchs_test), 
#                                 verbose=1,
#                                 steps = num_batchs_test)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])

