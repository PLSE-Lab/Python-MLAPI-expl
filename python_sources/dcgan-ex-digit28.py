#!/usr/bin/env python
# coding: utf-8

# * [keras-adversarial](https://github.com/bstriner/keras-adversarial)
# * [deep-learning-with-keras-ja/ch04/example_gan_convolutional.py](https://github.com/oreilly-japan/deep-learning-with-keras-ja/blob/master/ch04/example_gan_convolutional.py)

# In[ ]:


get_ipython().system('pip install keras==2.1.2')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from IPython import display
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import keras.backend as K
import numpy as np
from keras.layers import Input, Reshape


def dim_ordering_fix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 2, 3, 1))


def dim_ordering_unfix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 3, 1, 2))


def dim_ordering_shape(input_shape):
    if K.image_dim_ordering() == 'th':
        return input_shape
    else:
        return (input_shape[1], input_shape[2], input_shape[0])


def dim_ordering_input(input_shape, name):
    if K.image_dim_ordering() == 'th':
        return Input(input_shape, name=name)
    else:
        return Input((input_shape[1], input_shape[2], input_shape[0]), name=name)


def dim_ordering_reshape(k, w, **kwargs):
    if K.image_dim_ordering() == 'th':
        return Reshape((k, w, w), **kwargs)
    else:
        return Reshape((w, w, k), **kwargs)


def channel_axis():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return 3


# In[ ]:


get_ipython().system('pip install git+https://github.com/bstriner/keras-adversarial.git')


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras.backend as K
from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation, Dense, BatchNormalization, Conv2DTranspose
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling


# In[ ]:


def model_generator():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14)(g_input)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(int(nch / 2), (3, 3), padding="same")(H)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = Conv2D(int(nch / 4), (3, 3), padding="same")(H)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = Conv2D(1, (1, 1), padding="same")(H)
    g_V = Activation("sigmoid")(H)
    return Model(g_input, g_V)


# In[ ]:


# def model_generator():
#     nch = 512
#     g_input = Input(shape=[100])
#     H = Dense(7 * 7 * nch)(g_input)
#     H = Reshape((7, 7, nch))(H)
#     #H = BatchNormalization()(H)
#     H = Activation("relu")(H)
    
#     #H = dim_ordering_reshape(nch, 14)(H)
#     #H = UpSampling2D(size=(2, 2))(H)
#     H = Conv2DTranspose(filters=nch,
#                         kernel_size=3,
#                         strides=2,
#                         padding='same')(H)
#     #H = BatchNormalization()(H)
#     H = Activation("relu")(H)
#     H = Conv2DTranspose(filters=int(nch/2),
#                         kernel_size=3,
#                         strides=2,
#                         padding='same')(H)
#     #H = BatchNormalization()(H)
#     H = Activation("relu")(H)
#     H = Conv2DTranspose(filters=1,
#                         kernel_size=3,
#                         activation='sigmoid',
#                         padding='same',
#                         name='decoder_output')(H)
# #     H = Activation("relu")(H)
# #     H = Conv2D(1, (1, 1), padding="same")(H)
# #     g_V = Activation("sigmoid")(H)
#     return Model(g_input, H)


# In[ ]:


def model_discriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 512
    # nch = 128
    H = Conv2D(int(nch / 2), (5, 5),
               strides=(2, 2),
               padding="same",
               activation="relu",
               )(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(nch, (5, 5),
               strides=(2, 2),
               padding="same",
               activation="relu",
               )(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation="sigmoid")(H)
    return Model(d_input, d_V)


# In[ ]:


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


# In[ ]:


# z \in R^100
latent_dim = 100
# x \in R^{28x28}
input_shape = (1, 28, 28)


# In[ ]:


zsamples = np.random.normal(size=(10 * 10, latent_dim)) # fix zsamples
def generator_sampler(latent_dim, generator, zsamples=zsamples):
    def fun(zsamples=zsamples):
        #zsamples = np.random.normal(size=(10 * 10, latent_dim))
        gen = dim_ordering_unfix(generator.predict(zsamples))
        return gen.reshape((10, 10, 28, 28))
    return fun


# In[ ]:


# generator (z -> x)
generator = model_generator()
# discriminator (x -> y)
discriminator = model_discriminator(input_shape=input_shape)


# In[ ]:


generator.name = 'generator'
discriminator.name = 'discriminator'
generator.name, discriminator.name


# In[ ]:


gan0 = simple_gan(generator, discriminator, latent_sampling=None)
gan0.summary()


# In[ ]:


(32,) + (100,)


# In[ ]:


gan0.inputs


# In[ ]:


# gan (x - > yfake, yreal), z generated on GPU
gan = simple_gan(generator, discriminator,
                 normal_latent_sampling((latent_dim,)))


# In[ ]:


generator.summary()


# In[ ]:


discriminator.summary()


# In[ ]:


# print summary of models
gan.summary()


# In[ ]:


gan.inputs


# In[ ]:


gan.outputs


# In[ ]:


# build adversarial model
model = AdversarialModel(base_model=gan,
                         player_params=[generator.trainable_weights, 
                                        discriminator.trainable_weights],
                         player_names=["generator", "discriminator"])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                          player_optimizers=[Adam(1e-4, decay=1e-4),
                                             Adam(1e-3, decay=1e-4)],
                          loss="binary_crossentropy")


# In[ ]:


# train model
generator_cb = ImageGridCallback("output/gan_convolutional/epoch-{:03d}.png",
                                 generator_sampler(latent_dim, generator))
callbacks = [generator_cb]
if K.backend() == "tensorflow":
    callbacks.append(
        TensorBoard(log_dir=os.path.join("output/gan_convolutional/", "logs/"),
                    histogram_freq=0, write_graph=True, write_images=True))


# In[ ]:


callbacks


# In[ ]:


xtrain, xtest = mnist_data()
xtrain = dim_ordering_fix(xtrain.reshape((-1, 1, 28, 28)))
xtest = dim_ordering_fix(xtest.reshape((-1, 1, 28, 28)))
y = gan_targets(xtrain.shape[0])
ytest = gan_targets(xtest.shape[0])

len(y), y[0].shape, y[1].shape, y[2].shape, y[3].shape


# In[ ]:


y[0]


# In[ ]:


y[1]


# In[ ]:


y[2]


# In[ ]:


y[3]


# In[ ]:


history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), 
                    callbacks=callbacks, epochs=5,
                    batch_size=32)


# In[ ]:


ls -la output/gan_convolutional


# In[ ]:


display.display_png(display.Image('output/gan_convolutional/epoch-004.png'))


# In[ ]:


ls -la output/gan_convolutional/logs


# In[ ]:


res = generator.predict(zsamples)
res.shape


# In[ ]:


res.min(), res.max()


# In[ ]:


plt.imshow(res[0].reshape((28,28)))


# In[ ]:


discriminator.predict(res)


# In[ ]:


xtrain[0].shape


# In[ ]:


discriminator.predict(xtrain[:5])


# In[ ]:


y[0].shape, y[1].shape, y[2].shape, y[3].shape


# In[ ]:


y[3]


# In[ ]:




