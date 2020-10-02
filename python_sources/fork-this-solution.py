#!/usr/bin/env python
# coding: utf-8

# ## Please enter your name or nick name here

# In[ ]:


Participant_name = 'test' # inside the apostrophes e.g. Zel


# ## Enter your segmented file pathway here 
# ### (point them to input/SOMETHING/SOMETHING.tif)

# In[ ]:


label_pathway1 = '../input/testing/train1_label.tif'  # after the slash e.g. ../awesome/your_work1.tif


# In[ ]:


label_pathway2 = '../input/testing/train2_label.tif'  # after the slash e.g. ../awesome/your_work2.tif


# ## Play with some parameters

# In[ ]:


hyperparameters = {
    # network structural parameters
    'number_convolution_per_layer': 64,  
    'convolution_kernel_size': 3,
    'number_layer': 3,
    'dropout': 0.1,
    # experimental parameters
    'batch_size': 200,
    'number_epoch': 1, 
    'stride': 80,  # gap of sampling
    'width_sampling': 80,  #for window of sampling
}


# ## Import libraries and define tool functions

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing as mp
import cv2
from skimage.filters import hessian
from itertools import product, repeat
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, concatenate, UpSampling2D,     MaxPooling2D, BatchNormalization, Dropout, Input, Reshape, Flatten, Softmax


# ## Define some useful functions for data preparation and this competition

# In[ ]:


def shuffle(X, y):
    idx = np.random.permutation(X.shape[0])
    return X[idx], y[idx]

def patching(X, patch_size=256, stride=5):
    p_h = (X.shape[0] - patch_size) // stride + 1
    p_w = (X.shape[1] - patch_size) // stride + 1

    # stride the tensor
    _strides = tuple([i * stride for i in X.strides]) + tuple(X.strides)
    X = np.lib.stride_tricks.as_strided(X, shape=(p_h, p_w, patch_size, patch_size), strides=_strides)        .reshape((-1, patch_size, patch_size, 1))
    return X

def reconstruct_cnn(y_pred, image_size=None, stride=None):
    i_h, i_w = image_size[:2]  #e.g. (a, b)
    p_h, p_w = y_pred.shape[1:3]  #e.g. (x, h, w, 1)
    img = np.zeros((i_h, i_w))

    # compute the dimensions of the patches array
    n_h = (i_h - p_h) // stride + 1
    n_w = (i_w - p_w) // stride + 1

    for p, (i, j) in zip(y_pred, product(range(n_h), range(n_w))):
        img[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + stride, p_h, i_h - i) *
                               min(j + stride, p_w, i_w - j))
    return img


def _minmaxscalar(ndarray, dtype=np.float32):
    scaled = np.array((ndarray - np.min(ndarray)) / (np.max(ndarray) - np.min(ndarray)), dtype=dtype)
    return scaled


def submit(y_pred, idx, out_path):
    assert isinstance(y_pred, np.ndarray), 'y_pred should be a numpy array'
    assert isinstance(out_path, str), 'out_path should be a string'
    csv = {
        'id': idx,
        'prediction': y_pred.flatten().astype(int),
    }
    pd.DataFrame(csv).to_csv(out_path, index=False, header=True)


# ## Here to define the Convolutional Neural Network

# In[ ]:


def convolutionalNeuralNetwork(params):
    inputs = Input((params['width_sampling'], params['width_sampling'], 1))
    pools = []
    
    # encoder
    for i in range(params['number_layer']):
        if i == 0:
            x = inputs
        x = Conv2D(params['number_convolution_per_layer'],
                   (params['convolution_kernel_size'], params['convolution_kernel_size']),
                   activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(params['number_convolution_per_layer'], 
                   (params['convolution_kernel_size'], params['convolution_kernel_size']), 
                   activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        pools.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    x = Conv2D(params['number_convolution_per_layer'] * 4, 
               (params['convolution_kernel_size'], params['convolution_kernel_size']), 
               activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(params['number_convolution_per_layer'] * 4, 
               (params['convolution_kernel_size'], params['convolution_kernel_size']), 
               activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, 
               (params['convolution_kernel_size'], params['convolution_kernel_size']),
               activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # bottom
    x = Flatten()(x)
    for i in range(3):
        x = Dense(params['width_sampling'] * params['width_sampling'] // 64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout'])(x)
    x = Reshape((int(params['width_sampling'] // 8), int(params['width_sampling'] // 8), 1))(x)

    # decoder
    for i in range(params['number_layer']):
        x = Conv2DTranspose(params['number_convolution_per_layer'], 
                            (params['convolution_kernel_size'], params['convolution_kernel_size']), 
                            activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(params['number_convolution_per_layer'], 
                            (params['convolution_kernel_size'], params['convolution_kernel_size']), 
                            activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = concatenate([UpSampling2D(size=(2, 2))(x), pools[-i-1]], axis=3)

    x = Conv2DTranspose(params['number_convolution_per_layer'], 
                        (params['convolution_kernel_size'], params['convolution_kernel_size']), 
                        activation='relu', padding='same')(x)
    x = Conv2DTranspose(params['number_convolution_per_layer'], 
                        (params['convolution_kernel_size'], params['convolution_kernel_size']), 
                        activation='relu', padding='same')(x)

    x = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(x)

    mdl = Model(inputs=inputs, outputs=x)
    mdl.compile(loss='mse', optimizer='Adam')
    return mdl


# In[ ]:


raw1 = np.asarray(Image.open('../input/nanoperando-amiens-2019/train1.tif'))
raw2 = np.asarray(Image.open('../input/nanoperando-amiens-2019/train2.tif'))
to_predict1 = np.asarray(Image.open('../input/nanoperando-amiens-2019/test1.tif'))
to_predict2 = np.asarray(Image.open('../input/nanoperando-amiens-2019/test2.tif'))
label1 = np.asarray(Image.open(label_pathway1))
label2 = np.asarray(Image.open(label_pathway2))
idx = np.asarray(np.loadtxt('../input/nanoperando-amiens-2019/idx.csv')).astype(int)


#  ## Prepare data

# In[ ]:


X_train = patching(raw1, hyperparameters['width_sampling'], hyperparameters['stride'])
X_train = np.concatenate([X_train, patching(raw2, hyperparameters['width_sampling'], hyperparameters['stride'])], axis=0)
y_train = patching(label1, hyperparameters['width_sampling'], hyperparameters['stride'])
y_train = np.concatenate([y_train, patching(label2, hyperparameters['width_sampling'], hyperparameters['stride'])], axis=0)
X_train, y_train = shuffle(X_train, y_train)


# ## Begin training

# In[ ]:


model = convolutionalNeuralNetwork(hyperparameters)


# In[ ]:


# train
model.fit(X_train, y_train, epochs=hyperparameters['number_epoch'], batch_size=hyperparameters['batch_size'])
del X_train, y_train


# ## Prediction

# In[ ]:


X_test1 = patching(to_predict1, hyperparameters['width_sampling'], hyperparameters['stride'])
X_test2 = patching(to_predict2, hyperparameters['width_sampling'], hyperparameters['stride'])
y_pred1 = model.predict(X_test1)
y_pred1 = reconstruct_cnn(np.squeeze(y_pred1), image_size=to_predict1.shape, stride=hyperparameters['stride'])
y_pred2 = model.predict(X_test2)
y_pred2 = reconstruct_cnn(np.squeeze(y_pred2), image_size=to_predict2.shape, stride=hyperparameters['stride'])
total = np.concatenate([y_pred1.flatten(), y_pred2.flatten()], axis=0)
print('pred1 shape:{}'.format(y_pred1.shape), '\npred2 shape:{}'.format(y_pred2.shape),
      '\nidx shape:{}'.format(idx.shape))
submit(total.flatten()[idx], idx, '{}.csv'.format(Participant_name))


# ## Submit also other data/labels

# In[ ]:


from shutil import copyfile
copyfile(label_pathway1, '/kaggle/working/{}1.tif'.format(Participant_name))
copyfile(label_pathway2, '/kaggle/working/{}2.tif'.format(Participant_name))


# ## Plot the prediction

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(50, 50))
plt.imshow(y_pred1)


# In[ ]:


plt.figure(figsize=(50, 50))
plt.imshow(y_pred2)


# In[ ]:


## For the debug


# In[ ]:


print('test1.shape: {}'.format(to_predict1.shape))
print('test2.shape: {}'.format(to_predict2.shape))
print('y_pred1.shape: {}'.format(y_pred1.shape))
print('y_pred2.shape: {}'.format(y_pred2.shape))


# In[ ]:




