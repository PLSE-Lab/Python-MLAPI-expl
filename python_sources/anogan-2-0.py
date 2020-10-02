# %% [code]
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

# %% [code]
from collections import defaultdict


from PIL import Image
from six.moves import range

import matplotlib
import matplotlib.pyplot as plt

import datetime

from scipy.stats import truncnorm

import keras.backend as K

from keras.models import Model

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation, ZeroPadding2D

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.regularizers import l2
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.utils.generic_utils import Progbar

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# %% [code]
import matplotlib.pyplot as plt

# %% [code]

# TRAIN_IMG = os.listdir('/kaggle/input/steel-surface-defect-detection/train_images')
# TEST_IMG = os.listdir('/kaggle/input/steel-surface-defect-detection/test_images')
# print('Original train count =',len(TRAIN_IMG),', Original test count =',len(TEST_IMG))
# print('New train count = 1801 , New test count = 1801')
# os.mkdir('../tmp/')
# os.mkdir('../tmp/train_images/')
# r = np.random.choice(TRAIN_IMG,len(TEST_IMG),replace=False)
# for i,f in enumerate(r):
#     img = Image.open('../input/steel-surface-defect-detection/train_images/'+f)
#     img.save('../tmp/train_images/'+f)
# os.mkdir('../tmp/test_images/')
# for i,f in enumerate(TEST_IMG):
#     img = Image.open('../input/steel-surface-defect-detection/test_images/'+f)
#     img.save('../tmp/test_images/'+f)

# # %% [code]
# img_dir = '../tmp/'
# img_height = 256; img_width = 256
# batch_size = 32; nb_epochs = 15

# train_datagen = ImageDataGenerator(rescale=1./255,
#     horizontal_flip=True,
#     validation_split=0.2) # set validation split

# train_generator = train_datagen.flow_from_directory(
#     img_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training') # set as training data

# validation_generator = train_datagen.flow_from_directory(
#     img_dir, # same directory as training data
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation') # set as validation data

# %% [code]
import math

# %% [code]
def combine_images(generated_images): #for visualizing
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]), dtype = generated_images.dtype)
    
    for index, image in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image

# %% [code]
def generator_model():
    inputs = Input((10,))
    fc1 = Dense(input_dim=10, units=128*7*7)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((7, 7, 128), input_shape=(128*7*7,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    outputs = Activation('tanh')(conv2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# %% [code]
def discriminator_model():
    inputs = Input((28, 28, 1))
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    fc1 = Flatten()(pool2)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# %% [code]
#we'll train discriminator on generator model(for training generator)

def disc_on_gen(g, d):
    d.trainable = False
    gan_input = Input((10,))
    x = g(gan_input)
    gan_output = d(x)
    gan = Model(inputs = gan_input, outputs = gan_output)
    return gan
    

# %% [code]
from keras.optimizers import RMSprop
from keras import initializers
import tensorflow as tf
from tqdm import tqdm
import cv2

# %% [code]
def load_model():
    d = discriminator_model()
    g = generator_model()
    d_optim = RMSprop()
    g_optim = RMSprop(lr = 0.0002)
    g.compile(loss = 'binary_crossentropy', optimizer = g_optim)
    d.compile(loss = 'binary_crossentropy', optimizer = d_optim)
    g.load_weights('./weights/discriminator.h5')
    d.load_weights('./weights/generator.h5')
    return g, d

# %% [code]
def train(BATCH_SIZE, X_train):
    
    ### model define
    d = discriminator_model()
    g = generator_model()
    d_on_g = disc_on_gen(g, d)
    d_optim = RMSprop(lr=0.0004)
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='mse', optimizer=g_optim)
    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)
    

    for epoch in range(10):
        print ("Epoch is", epoch)
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):
            # create random noise -> U(0,1) 10 latent vectors
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 10))

            # load real data & generate fake data
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            
            # visualize training results
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                cv2.imwrite('./result/'+str(epoch)+"_"+str(index)+".png", image)

            # attach label for training discriminator
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            
            # training discriminator
            d_loss = d.train_on_batch(X, y)

            # training generator
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True

            progress_bar.update(index, values=[('g',g_loss), ('d',d_loss)])
        print ('')

        # save weights for each epoch
        g.save_weights('weights/generator.h5', True)
        d.save_weights('weights/discriminator.h5', True)
    return g, d

# %% [code]
def generate(BATCH_SIZE):  #generating images
    g = generator_model()
    g.load_weights('./weights/generator.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 10))
    generated_images = g.predict(noise)
    return generated_images

# %% [code]
#anomaly loss function
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

# %% [code]
#discriminator intermediate layer features extraction
def feature_extractor(d = None):
    if d is None:
        d = discriminator_model()
        d.load_weights('./weights/discriminator.h5')
    intermediate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermediate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermediate_model

# %% [code]
def anomaly_detector(g=None, d=None):    #defining anomaly detector
    if g is None:
        g = generator_model()
        g.load_weights('weights/generator.h5')
    intermediate_model = feature_extractor(d)
    intermediate_model.trainable = False
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    
    # Input layer can't be trained. Add new layer as same size & same distribution
    
    aInput = Input(shape=(10,))
    gInput = Dense((10), trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)
    
    # G & D feature
    G_out = g(gInput)
    D_out= intermediate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.90, 0.10], optimizer='rmsprop')
    
    # batchnorm learning phase fixed (test) : make non trainable
    K.set_learning_phase(0)
    
    return model

# %% [code]
# anomaly detection
def compute_anomaly_score(model, x, iterations=500, d=None):
    z = np.random.uniform(0, 1, size=(1, 10))
    
    intermediate_model = feature_extractor(d)
    d_x = intermediate_model.predict(x)

    # learning for changing latent
    loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)
    similar_data, _ = model.predict(z)
    
    loss = loss.history['loss'][-1]
    
    return loss, similar_data