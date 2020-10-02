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


from __future__ import absolute_import, print_function, division

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[ ]:


# import dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(f'training data shape: {train_images.shape}')
print(f'testing data shape: {test_images.shape}')


# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sns.countplot(train_labels)


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[ ]:


sample_train_img = train_images[0]
sample_test_img = test_images[0]

fig = plt.figure(0, (12, 4))

ax1 = plt.subplot(1,2,1)
ax1.imshow(np.array(sample_train_img).reshape(28, 28), cmap='gray')
ax1.set_title('train sample image')

ax2 = plt.subplot(1,2,2)
ax2.imshow(np.array(sample_test_img).reshape(28, 28), cmap='gray')
ax2.set_title('test sample image')

plt.show()


# In[ ]:


def image_preprocessing(img):
    img = img / 255.
    return img.reshape(-1, 28, 28, 1)

train_images = image_preprocessing(train_images)
test_images = image_preprocessing(test_images)


# In[ ]:


class AutoEncoder():
    def __init__(self, input_shape, latent_dim=64, net_type='mlp', print_summary=True, **kwargs):
        self.input_shape = (
            input_shape
            if len(input_shape) == 3 else
            (*input_shape, 1)
        )
        self.latent_dim = latent_dim
        self.net_type = net_type
        self.print_summary = print_summary
        self.is_trained = False
        
        self._validate_params(kwargs)
        
        self.autoencoder = self._stack_neural_nets()
        
        if self.print_summary:
            print('\n', self.autoencoder.summary())

    def _validate_params(self, kwargs):
        if self.input_shape[2] > 3:
            raise ValueError(f'Max supported channels are 3 but got {self.input_shape[2]}')

        if self.net_type not in ['mlp', 'cnn']:
            raise ValueError(f'`net_type` should be MLP or CNN but got {self.net_type}')

        if 'initializer' not in kwargs:
            self.initializer = keras.initializers.glorot_normal(seed=42)
        else:
            if isinstance(kwargs['initializer'], keras.initializers.Initializer):
                self.initializer = kwargs['initializer']
            else:
                raise ValueError(f'''`initializer` should be of type keras.initializer 
                but got {type(kwargs['initializer'])}''')
    
    def _build_layers_dims(self):
        self.layers_dims = []
        input_layer_dim = self.input_shape[0] * self.input_shape[1]
        for i in range(2):
            self.layers_dims.append(input_layer_dim // pow(2, i+1))
            
    def _loss(self):
        pass

    def _encoder_mlp(self):
        encoder = Sequential(name='Encoder')
        
        #input layer
        encoder.add(Flatten(input_shape=self.input_shape, name='input_layer'))
        
        #hidden layers
        for i, units in enumerate(self.layers_dims):
            encoder.add(Dense(units=units, kernel_initializer=self.initializer, name=f'hidden_layer_{i+1}'))
            encoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_{i+1}'))
            encoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_{i+1}'))

        #latten layer
        encoder.add(Dense(units=self.latent_dim, kernel_initializer=self.initializer, name='latent_layer'))
        encoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_latent_layer'))
        encoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_latent_layer'))

        if self.print_summary:
            print('\n', encoder.summary())
        
        encoder_input = Input(shape=self.input_shape)
        encoder_output = encoder(encoder_input)
        
        return Model(inputs=encoder_input, outputs=encoder_output)

    def _encoder_cnn(self):
        encoder = Sequential(name='Encoder')
        
        #convolutional layers
        encoder.add(Conv2D(
            filters=1, kernel_size=(3,3), padding='same', kernel_initializer=self.initializer,
            input_shape=self.input_shape, name='convolutional_1'
        ))
        encoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_1'))
        encoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_1'))


        encoder.add(Conv2D(
            filters=32, kernel_size=(3,3), padding='same',kernel_initializer=self.initializer,
            input_shape=self.input_shape, name='convolutional_2'
        ))
        encoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_2'))
        encoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_2'))
        encoder.add(MaxPooling2D(pool_size=(2,2), name='maxpooling_2'))

        encoder.add(Conv2D(
            filters=16, kernel_size=(3,3), padding='same',
            kernel_initializer=self.initializer, name='convolutional_3'
        ))
        encoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_3'))
        encoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_3'))
        encoder.add(MaxPooling2D(pool_size=(2,2), name='maxpooling_3'))

        encoder.add(Flatten(name='flattend_layer'))
        
        #hidden layers
        for i, units in enumerate(self.layers_dims):
            encoder.add(Dense(units=units, kernel_initializer=self.initializer, name=f'hidden_layer_{i+1}'))
            encoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_{i+4}'))
            encoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_{i+4}'))

            
        #latten layer
        encoder.add(Dense(units=self.latent_dim, kernel_initializer=self.initializer, name='latent_layer'))
        encoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_latent_layer'))
        encoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_latent_layer'))

        
        if self.print_summary:
            print('\n', encoder.summary())
        
        encoder_input = Input(shape=self.input_shape)
        encoder_output = encoder(encoder_input)
        
        return Model(inputs=encoder_input, outputs=encoder_output)

    def _decoder_mlp(self):
        decoder = Sequential(name='Decoder')

        #input layer
        decoder.add(Dense(units=self.layers_dims[-1], input_dim=self.latent_dim,
                          kernel_initializer=self.initializer, name='hidden_layer_1'))
        decoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_1'))
        decoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_1'))

        
        #hidden layers
        for i, units in enumerate(self.layers_dims[::-1][1:]):
            decoder.add(Dense(units=units, kernel_initializer=self.initializer, name=f'hidden_layer_{i+2}'))
        decoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_{i+2}'))
        decoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_{i+2}'))
        
        #output layer
        decoder.add(Dense(units=self.input_shape[0]*self.input_shape[1],
                          activation='sigmoid', kernel_initializer=self.initializer, name='output_layer'))
        decoder.add(Reshape(self.input_shape, name='reshaped_output_layer'))

        if self.print_summary:
            print('\n', decoder.summary())
        
        decoder_input = Input(shape=(self.latent_dim,))
        decoder_output = decoder(decoder_input)
        
        return Model(inputs=decoder_input, outputs=decoder_output)

    def _decoder_cnn(self):
        decoder = Sequential(name='Decoder')

        #hidden layers
        decoder.add(Dense(units=self.layers_dims[-1], input_dim=self.latent_dim,
                          kernel_initializer=self.initializer, name='hidden_layer_1'))
        decoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_1'))
        decoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_1'))

        for i, units in enumerate(self.layers_dims[::-1][1:]):
            decoder.add(Dense(units=units, kernel_initializer=self.initializer, name=f'hidden_layer_{i+2}'))
            decoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_layer_{i+2}'))
            decoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_layer_{i+2}'))
        
        decoder.add(Dense(units=self.input_shape[0]*self.input_shape[1], activation='sigmoid',
                          kernel_initializer=self.initializer, name='output_layer'))
        decoder.add(Reshape((7, 7, 16), name='reshaping_layer'))

        #Deconvolutional layers
        decoder.add(Conv2DTranspose(
            filters=16, kernel_size=(3,3), padding='same', name='deconvolutional_1'
        ))
        decoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_convlayer_1'))
        decoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_convlayer_1'))
        decoder.add(UpSampling2D(size=(2,2), name='upsampling_1'))
        
        decoder.add(Conv2DTranspose(
            filters=32, kernel_size=(3,3), padding='same', name='deconvolutional_2'
        ))
        decoder.add(LeakyReLU(alpha=0.2, name=f'leakyrelu_convlayer_2'))
        decoder.add(BatchNormalization(momentum=0.8, name=f'batchnorm_convlayer_2'))
        decoder.add(UpSampling2D(size=(2,2), name='upsampling_2'))

        decoder.add(Conv2DTranspose(
            filters=1, kernel_size=(3,3), padding='same',
            activation='sigmoid', name='deconvolutional_3'
        ))
        
        if self.print_summary:
            print('\n', decoder.summary())
        
        decoder_input = Input(shape=(self.latent_dim,))
        decoder_output = decoder(decoder_input)
        
        return Model(inputs=decoder_input, outputs=decoder_output)
    
    def _stack_neural_nets(self):
        self._build_layers_dims()
        
        nn_input = Input(self.input_shape)
        
        self.encoder = (
            self._encoder_mlp()
            if self.net_type == 'mlp' else
            self._encoder_cnn()
        )
        encoded_img = self.encoder(nn_input)
        
        self.decoder = (
            self._decoder_mlp()
            if self.net_type == 'mlp' else
            self._decoder_cnn()
        )
        decoded_img = self.decoder(encoded_img)
        
        return Model(inputs=nn_input, outputs=decoded_img, name='Autoencoder')

    def train(self, X_train, X_valid=None, optim=Adam(), batch_size=64, epochs=20,
              train_if_trained=False, warm_start=False, plot_metric=True):
        if not train_if_trained:
            if self.is_trained:
                print('Already trained, if you further want to train set `train_if_trained` to Train')
                return
        
        if not warm_start and self.is_trained:
            self.autoencoder = self._stack_neural_nets()
            print('\nBegan training from start...\n')
        else:
            print('\nResumed training from previous run...\nif this is first run than it has no effect')

        if not isinstance(optim, keras.optimizers.Optimizer):
            raise ValueError(f'`optim` should be of type keras.optimizers but got {type(optim)}')

        if self.input_shape != X_train.shape[1:]:
            raise ValueError(f'''Training images should be of shape {self.input_shape} but
            got {X_train.shape[1:]}''')

        if self.input_shape != X_valid.shape[1:]:
            raise ValueError(f'''Validation images should be of shape {self.input_shape} but
            got {X_valid.shape[1:]}''')

        self.autoencoder.compile(
            loss= 'msle', #'binary_crossentropy',
            optimizer=optim
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            min_delta=0.00025,
            baseline=0.05,
            patience=4,
            restore_best_weights=True,
            verbose=10
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            patience=2,
            factor=0.15,
            min_delta=0.00025,
            min_lr=0,
            verbose=10
        )
        
        self.callbacks = [
            early_stopping,
            lr_scheduler
        ]

        self.history = self.autoencoder.fit(
            x=X_train,
            y=X_train,
            validation_data=(X_valid, X_valid),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
            use_multiprocessing=True
        )
        
        self.is_trained = True
        
        if plot_metric:
            self._plot_metrics()
    
    def _plot_metrics(self):
        fig = plt.figure(0, (8, 4))

        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(self.history.epoch, self.history.history['loss'])
        ax1.plot(self.history.epoch, self.history.history['val_loss'])
        ax1.set_title('train v/s validation loss')
        ax1.legend(['train', 'validation'])
        
    def compress(self, image):
        return self.encoder.predict(image)

    def decompress(self, latent_repr):
        return self.decoder.predict(latent_repr)
    
    def save(self, path, model_name='model'):
        if self.is_trained:
            model_path = os.path.join(path+model_name+'.h5')
            self.autoencoder.save(model_path)
        else:
            ValueError('Model is not trained yet, first train it by calling train() on AutoEncoder instance')


# In[ ]:


ae_mlp = AutoEncoder(input_shape=(28,28), net_type='mlp')


# In[ ]:


ae_mlp.train(X_train=train_images, X_valid=test_images[:9990], epochs=30)


# In[ ]:


ae_mlp.save(path='', model_name='ae_mlp_v01')


# In[ ]:


ae_cnn = AutoEncoder(input_shape=(28,28), net_type='cnn')


# In[ ]:


ae_cnn.train(X_train=train_images, X_valid=test_images[:9990], epochs=30)


# In[ ]:


ae_cnn.save(path='', model_name='ae_cnn_v01')


# In[ ]:


print('''COl_1: Original(28x28)
COl_2: MLP Compressed(8x8)
COl_3: MLP Decompressed(28x28)
COl_4: CNN Compressed(8x8)
COl_5: CNN Decompressed(28x28)''')

fig = plt.figure(0, (12, 18))

count = 1
for i in range(10):
    sample_test_img = test_images[9990+i: 9990+i+1]
    ax1 = plt.subplot(10, 5, count)
    count += 1
    ax1.imshow(sample_test_img.reshape((28,28)), cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])

    latent_repr = ae_mlp.compress(sample_test_img)
    latent_img = latent_repr.reshape(8, 8)
    decompressed_img = ae_mlp.decompress(latent_repr).reshape((28,28))
    
    ax2 = plt.subplot(10, 5, count)
    count += 1
    ax2.imshow(latent_img, cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = plt.subplot(10, 5, count)
    count += 1
    ax3.imshow(decompressed_img, cmap='gray')
    ax3.set_xticks([])
    ax3.set_yticks([])

    latent_repr = ae_cnn.compress(sample_test_img)
    latent_img = latent_repr.reshape(8, 8)
    decompressed_img = ae_cnn.decompress(latent_repr).reshape((28,28))

    ax4 = plt.subplot(10, 5, count)
    count += 1
    ax4.imshow(latent_img, cmap='gray')
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = plt.subplot(10, 5, count)
    count += 1
    ax5.imshow(decompressed_img, cmap='gray')
    ax5.set_xticks([])
    ax5.set_yticks([])

plt.show()


# In[ ]:




