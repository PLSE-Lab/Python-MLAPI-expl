#!/usr/bin/env python
# coding: utf-8

# The objective of this kernel is to train a simple CNN model to predict gender from face images. The training dataset consists of 29437 RGB images of human faces of different ages with 128 by 128 resolution. The network architecture includes a single layer of convolution with padding, local normalization, max pooling, and the relu activation. Mini-batch Adam optimizer is used to minimize the cross-entropy cost function. The model performance on the training and the test datasets along with the CNN architecture are provided at the end of the kernel. First, data are fetched from different folders.

# In[ ]:


import os
import matplotlib.pyplot as pyp
import numpy


# In[ ]:


def dataloader( train_or_val ):    
    X_cnn = []
    X_flat = []
    Y = []
    if train_or_val == 'T':
        path = "../input/fliker-face-gender/aligned/"
    elif train_or_val == 'V':
        path = "../input/gender-face-validation/valid/"
    folder_list = sorted(os.listdir(path))
    for folder in folder_list:
        img_name_list = sorted(os.listdir(path+folder))
        for img_name in img_name_list:
            ax = pyp.imread(path+folder+'/'+img_name)
            X_cnn.append(ax)
            X_flat.append(ax.flatten())
            if folder[3] == 'F':
                Y.append([1])
            elif folder[3] == 'M':
                Y.append([0])
    X_cnn = numpy.array(X_cnn)
    X_flat = numpy.array(X_flat)
    Y = numpy.array(Y)    
    m = Y.shape[0]    
    permutation = list(numpy.random.permutation(m))
    X_cnn = X_cnn[permutation,:]
    X_flat = X_flat[permutation,:]
    Y = Y[permutation,:]    
    return X_flat, X_cnn, Y 


# In[ ]:


X_train_flattened, X_train_cnn, Y_train = dataloader('T')
X_val_flattened, X_val_cnn, Y_val = dataloader('V')


# In[ ]:


print("number of training examples = " + str(X_train_cnn.shape[0]))
print("number of test examples = " + str(X_val_cnn.shape[0]))
print('\nTrain datasets shape')
print(X_train_cnn.shape,'<-- Dataset format for fitting a CNN')
print(X_train_flattened.shape,'<-- Dataset format for fitting a fully connected')
print(Y_train.shape,'<-- Target variable')
print('\nValidation datasets shape:\n')
print(X_val_cnn.shape,'<-- Dataset format for fitting a CNN')
print(X_val_flattened.shape,'<-- Dataset format for fitting a fully connected')
print(Y_val.shape,'<-- Target variable')


# In[ ]:


index = 135
pyp.imshow(X_train_cnn[index])
print ("sample image y = " + str(numpy.squeeze(Y_train[index,:])))


# In[ ]:


from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
K.set_image_data_format('channels_last')


# In[ ]:


def GenderModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='GenderModel')
    return model


# In[ ]:


GenderModel = GenderModel(X_train_cnn.shape[1:])
GenderModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
GenderModel.fit(X_train_cnn, Y_train, epochs=20, batch_size=16)


# In[ ]:


preds = GenderModel.evaluate(X_val_cnn, Y_val, batch_size=16, verbose=1, sample_weight=None)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[ ]:


GenderModel.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'


# In[ ]:


GenderModel.summary()


# In[ ]:


plot_model(GenderModel, to_file='GenderModel.png')

