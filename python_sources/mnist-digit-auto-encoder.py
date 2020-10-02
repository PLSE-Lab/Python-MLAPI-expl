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


# Basic imports for visualization

get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img

warnings.filterwarnings("ignore")


# In[ ]:


# Importing nueral network libraries - Keras

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Conv2D, Flatten)
from keras.layers import (Dense, Input)
from keras.layers import (Reshape, Conv2DTranspose)
from keras.models import Model
from keras.utils import plot_model
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.preprocessing import image


# In[ ]:


# loading the training data

train_df = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


# loading the test data

test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


test.head()


# In[ ]:


# creating the train and target dataset - from the trainig data

train = train_df.drop(['label'], axis=1)
target = train_df['label']


# In[ ]:


train_df.head()


# In[ ]:


target.head()


# In[ ]:


# displaying the target distribution

sns.countplot(target)


# In[ ]:


# reshaping the training, testing data and target

train = train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
target = np_utils.to_categorical(target, 10)


# In[ ]:


# dividing the data into traing and validation

import sklearn
from sklearn import model_selection

X_train, X_val, y_train, y_val = model_selection.train_test_split(train, target, test_size=0.20, random_state=1234567890)


# In[ ]:


print("Number of images in X_train: {}".format(X_train.shape))
print("Number of images in y_train: {}".format(y_train.shape))


# In[ ]:


print("Number of images in X_train: {}".format(X_val.shape))
print("Number of images in y_train: {}".format(y_val.shape))


# In[ ]:


# creating the AutoEncoder class

class AutoEncoder:

    # Initialisation of data
    def __init__(self, X_train, y_train, X_test, y_test, latent_dimension=16):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # network parameters
        self.batch_size = 128
        self.kernel_size = 3
        self.latent_dimension = latent_dimension

        # number of filters per Convolutional layer
        self.filters = [32, 64]

    # Normalizing the data
    def normalize(self):
        image_shape = self.X_train.shape[1]
        #self.X_train = np.reshape(self.X_train, [-1, image_shape, image_shape, 1])
        #self.X_test = np.reshape(self.X_test, [-1, image_shape, image_shape, 1])
        self.X_train = self.X_train.astype(np.float32) / 255
        self.X_test = self.X_test.astype(np.float32) / 255
        return image_shape
    

    # Building an encoder model
    def encoder(self, image_shape):

        # creating the inputs for the encoder model
        inputs = Input(shape=(image_shape, image_shape, 1), name="encoder-input")
        x = inputs

        # stacking the Conv2D(32 filters) and Conv2D(64 filters)
        for ftr in self.filters:
            x = Conv2D(filters=ftr,
                       kernel_size=self.kernel_size,
                       activation="relu",
                       strides=2,
                       padding="same")(x)
        shape = K.int_shape(x)

        # generating the latent vector
        x = Flatten()(x)
        latent = Dense(self.latent_dimension, name="latent-vector")(x)

        # initializing the encoder model
        encoderModel = Model(inputs, latent, name="encoder")
        encoderModel.summary()
        plot_model(encoderModel, to_file="encoder.png", show_shapes=True)

        # returning the shape which is used in the decoder
        return shape, inputs, encoderModel


    # Building an decoder model
    def decoder(self, shape):

        # creating the input for decoder model
        inputs = Input(shape=(self.latent_dimension,), name="decoder-input")

        # using the same shape which given by encoder
        x = Dense(shape[1] * shape[2] * shape[3])(inputs)

        # transforming the latent vector to stable shape (i.e., input shape)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # staking the transpose Conv2D(64) and Conv2D(32)
        for ftr in self.filters[::-1]:
            x = Conv2DTranspose(filters=ftr,
                                kernel_size=self.kernel_size,
                                activation="relu",
                                strides=2,
                                padding="same")(x)

        # reconstructing the input
        output = Conv2DTranspose(filters=1,
                                 kernel_size=self.kernel_size,
                                 activation="sigmoid",
                                 padding="same",
                                 name="decoder-output")(x)

        # initializing the decoder model
        decoderModel = Model(inputs, output, name="decoder")
        decoderModel.summary()
        plot_model(decoderModel, to_file="decoder.png", show_shapes=True)
        return decoderModel
    
    
    # Building the Auto Encoder Model
    def autoencoder(self, inputs, encoderModel, decoderModel):
        # auto-encoder = encoder + decoder
        autoencoderModel = Model(inputs, decoderModel(encoderModel(inputs)), name="auto-encoder")

        autoencoderModel.summary()
        plot_model(autoencoderModel, to_file="auto-encoder.png", show_shapes=False)
        return autoencoderModel    
    
    
    # Exponential Decay function - utility functions
    @staticmethod
    def exponential_decay_fn(epoch):
        return 0.01 * 0.1 ** (epoch / 20)

    @staticmethod
    def exponential_decay(lr, s):
        def exponential_decay_fn(epoch):
            return lr * 0.1 ** (epoch / s)

        return exponential_decay_fn    
    
    def callbacks(self):
        current_dt_time = datetime.datetime.now()
        model_name = 'model_init' + '_' + str(current_dt_time).replace(' ', '').replace(':', '_') + '/'
        filePath = model_name

        if not os.path.exists(model_name):
            os.mkdir(model_name)

        file_path = model_name + "model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}"                                  "-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5"

        # check point to save best model only
        checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_categorical_accuracy', verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)

        # check point avoid plateau
        LR = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.000001, verbose=1, cooldown=1)

        # check point to exponential decay for learning rate
        exponential_decay_fn = self.exponential_decay(lr=0.0001, s=20)
        lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
        callbacks = [checkpoint, lr_scheduler, LR]
        return (callbacks, filePath)     
    
    
    def train(self, autoEncoder, epochs, callbacks):
        # fitting the network
        history = autoEncoder.fit(self.X_train, self.X_train, validation_data=(self.X_test, self.X_test),
                                         epochs=epochs,
                                         verbose=1,
                                         batch_size=self.batch_size) 
        predicted = autoEncoder.predict(self.X_test)
        
        # Visualizing the data
        # displaying the 1st 8 digits input and predicted images
        images = np.concatenate([self.X_test[:8], predicted[:8]])
        images = images.reshape((4, 4, image_shape, image_shape))
        images = np.vstack([np.hstack(i) for i in images])
        
        # plotting
        plt.figure()
        plt.axis("off")
        plt.title("Input: 1st 2nd rows; predicted: last 2nd rows")
        plt.imshow(images, interpolation='none', cmap='gray')
        plt.show()
        
        return history
        


    def best_model(self, model_name):
        values = {}

        models = os.listdir(model_name)
        for model in models:
            converted = model.replace(".h5", "")
            accuracy = float(converted.split("-")[-1])
            values.update({accuracy: model})

        key = max(values, key=values.get)
        best = values.get(key)

        return best 
    
    # Function to visualisation of the MNIST digits distribution over 2D latent codes
    def plot_results(self, models, batch_size=32, name="auto-encoder-two-dimensions"):

        (encoderModel, decoderModel) = models
        os.makedirs(name, exist_ok=True)
        file_name = os.path.join(name, "latent-2D-image.png")

        # displaying 2D plot of the digit classes in the latent space
        z = encoderModel.predict(self.X_test, batch_size=batch_size)
        converted = np.argmax(z, axis=1)
        resulted = pd.Series(converted)

        # plotting the latent
        plt.figure(figsize=(10, 8))
        plt.scatter(z[:, 0], z[:, 1], c=converted)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig("filename")
        plt.show()

        # Digits over Latent
        file_name = os.path.join(name, "digits-over-latent.png")
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = np.array([[xi, yi]])
                decoded = decoderModel.predict(z)
                digit = decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(20, 20))
        start = digit_size // 2
        end = n * digit_size + start + 1
        pixel = np.arange(start, end, digit_size)
        sample_x = np.round(grid_x, 1)
        sample_y = np.round(grid_y, 1)
        plt.xticks(pixel, sample_x)
        plt.yticks(pixel, sample_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.savefig(file_name)
        plt.show()        


# In[ ]:


# initializing the auto encoder instance

autoEncoder = AutoEncoder(X_train, y_train, X_val, y_val)


# In[ ]:


# Normalizing the data

image_shape = autoEncoder.normalize()


# In[ ]:


# Creating the encoder Model

shape, inputs, encoderModel = autoEncoder.encoder(image_shape)


# In[ ]:


# Displaying the encoder network

plt.figure(figsize=(8, 10))
encoderImage = img.imread("encoder.png")
imagePlot = plt.imshow(encoderImage)
plt.show()


# In[ ]:


# Creating the decoder model

decoderModel = autoEncoder.decoder(shape)


# In[ ]:


# Displaying the decoder network

plt.figure(figsize=(8, 10))
decoderImage = img.imread("decoder.png")
imagePlot = plt.imshow(decoderImage)
plt.show()


# In[ ]:


# Creating an auto encoder model

autoEnc = autoEncoder.autoencoder(inputs, encoderModel, decoderModel)


# In[ ]:


# Displaying the autoencoder network

plt.figure(figsize=(8, 10))
autoEncoderImage = img.imread("auto-encoder.png")
imagePlot = plt.imshow(autoEncoderImage)
plt.show()


# In[ ]:


# Compiling the model
# Loss function: Mean Square Error
# Optimizer: RMSProp

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)

autoEnc.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['categorical_accuracy'])


# In[ ]:


# getting call backs and model paths

(callbacks, model_name) = autoEncoder.callbacks()


# In[ ]:


history = autoEncoder.train(autoEnc, 50, callbacks)


# In[ ]:


history.history.keys()


# In[ ]:


plt.plot(history.history['categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


# plotting the result

autoEncoder = AutoEncoder(X_train, y_train, X_val, y_val, 2)
image_shape = autoEncoder.normalize()
shape, inputs, encoderModel = autoEncoder.encoder(image_shape)
decoderModel = autoEncoder.decoder(shape)


# In[ ]:


models = (encoderModel, decoderModel)
autoEncoder.plot_results(models)


# In[ ]:




