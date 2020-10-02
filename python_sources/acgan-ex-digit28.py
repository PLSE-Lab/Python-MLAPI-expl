#!/usr/bin/env python
# coding: utf-8

# * [keras-adversarial](https://github.com/bstriner/keras-adversarial)
# * https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras.backend as K
from tqdm import tqdm

from keras.layers import (Flatten, Dropout, LeakyReLU, Input, Activation, Dense, BatchNormalization,
                          Embedding, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D)
from keras.layers import Lambda, concatenate, average, multiply
from keras.layers import Input, Reshape, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import to_categorical


# In[ ]:





# In[2]:


(X_train, y_train), (_, _) = mnist.load_data()
print(X_train.max())
X_train = (X_train.astype(np.float32)-255.0/2) / (255.0/2)
print(X_train[0].shape)


# In[3]:


x_train = X_train.reshape((-1, 28, 28, 1))
y_train


# In[4]:


# z \in R^100
latent_dim = 100
# x \in R^{28x28}
input_shape = (28, 28, 1)


# In[5]:


# def model_generator(num_classes=10):
#     nch = 100
#     label = Input(shape=(1,), dtype='int32', name='label')
#     label_embedding = Flatten()(Embedding(num_classes, nch)(label))
    
#     g_input = Input(shape=[nch], name='noise')
    
#     inp = multiply([g_input, label_embedding])
    
#     H = Reshape((1,1,nch))(inp)
#     H = Conv2DTranspose(filters=512, kernel_size=(3, 3))(H)
#     H = BatchNormalization()(H)
#     H = Activation("relu")(H)
#     H = Conv2DTranspose(filters=256, kernel_size=(3, 3))(H)
#     H = BatchNormalization()(H)
#     H = Activation("relu")(H)
#     H = Conv2DTranspose(filters=128, kernel_size=(3, 3))(H)
#     H = BatchNormalization()(H)
#     H = Activation("relu")(H)
#     H = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(H)
#     H = BatchNormalization()(H)
#     H = Activation("relu")(H)
#     H = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2))(H)
#     H = BatchNormalization()(H)
#     H = Activation("relu")(H)
#     H = Conv2D(1, (1, 1), padding="same")(H)
#     g_V = Activation("sigmoid")(H)
#     return Model([g_input, label], g_V, name='generator')

# generator = model_generator()
# generator.summary()


# In[6]:


def model_generator(latent_dim=latent_dim, num_classes=10):

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)

generator = model_generator()
generator.summary()


# In[7]:


# def model_discriminator(input_shape=(28, 28, 1), dropout_rate=0.5, num_classes=10):
#     d_input = Input(input_shape, name="input_img")
    
#     oup_ae = d_input
#     oup_ae = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='elu')(oup_ae)
#     oup_ae = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='elu')(oup_ae)
#     oup_ae = MaxPooling2D(pool_size=(2, 2))(oup_ae)
#     #oup_ae = Dropout(0.25)(oup_ae)
#     oup_ae = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='elu')(oup_ae)
#     oup_ae = Conv2D(filters=64, kernel_size=(3,3), padding='same')(oup_ae)
    
#     oup_ae1 = GlobalMaxPooling2D()(oup_ae)
#     oup_ae2 = GlobalAveragePooling2D()(oup_ae)
#     oup = concatenate([oup_ae1, oup_ae2])
#     #model_cnvt_org = Model(inp, oup, name='model_img_cnvt_org')
#     # oup1 = Activation('sigmoid')(oup)
#     oup1 = BatchNormalization()(oup)

# #     H = Conv2D(int(nch / 2), (5, 5),
# #                strides=(2, 2),
# #                padding="same",
# #                activation="relu",
# #                )(d_input)
# #     H = LeakyReLU(0.2)(H)
# #     H = Dropout(dropout_rate)(H)
# #     H = Conv2D(nch, (5, 5),
# #                strides=(2, 2),
# #                padding="same",
# #                activation="relu",
# #                )(H)
# #     H = LeakyReLU(0.2)(H)
# #     H = Dropout(dropout_rate)(H)
# #     H = Flatten()(H)
# #     H = Dense(int(nch / 2))(H)
# #     H = LeakyReLU(0.2)(H)
# #     H = Dropout(dropout_rate)(H)
    
#     validity = Dense(1, activation="sigmoid", name='validity')(oup1)
#     labels = Dense(num_classes, activation="softmax", name='labels')(oup1)
#     return Model(d_input, [validity, labels], name='discriminator')

# # discriminator = model_discriminator(input_shape=input_shape)
# # discriminator.summary()


# In[22]:


def model_discriminator(img_shape=(28,28,1), num_classes=10):

    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    img = Input(shape=img_shape)

    # Extract feature representation
    features = model(img)

    # Determine validity and label of the image
    validity = Dense(1, activation="sigmoid", name='validity')(features)
    label = Dense(num_classes, activation="softmax", name='labels')(features)

    return Model(img, [validity, label])

discriminator = model_discriminator()
discriminator.summary()


# In[9]:


# generator (z -> x)
generator = model_generator()
generator.summary()


# In[10]:


# discriminator (x -> y)
discriminator = model_discriminator()
discriminator.summary()


# In[23]:


def make_models(generator, discriminator):
    z = generator.inputs[0]
    print(z)
    label = generator.inputs[1]
    print(label)
    inp = discriminator.inputs[0]
    
    # Generator
    model_g = generator
    
    # Discriminator
    model_d = Model(inp, discriminator(inp), name='discriminator')
    model_d.compile(optimizer=Adam(0.0002, 0.5), loss=['binary_crossentropy', 'categorical_crossentropy'])
    
    # Generator -> Discriminator
    img_fake = generator([z, label])
    discriminator.trainable = False
    oup_gd = discriminator(img_fake)
    #oup_gd = Activation('linear', name='gd_valid')(oup_gd)
    model_gd = Model([z, label], oup_gd, name='model_gd')
    model_gd.compile(optimizer=Adam(0.0002, 0.5), loss=['binary_crossentropy', 'categorical_crossentropy'])
    
    return {
        'model_d': model_d,
        'model_g': model_g,
        'model_gd': model_gd,
    }

generator = model_generator()
discriminator = model_discriminator()
#generator.name = 'generator'
#discriminator.name = 'discriminator'
print(generator.name, discriminator.name)

models = make_models(generator, discriminator)
models


# In[24]:


models['model_gd'].summary()


# In[25]:


models['model_gd'].predict([np.random.normal(0, 1, (5,100)), np.array([0,1,2,3,4])])


# In[26]:


models['model_g'].predict([np.random.normal(0, 1, (3,100)), np.array([0,1,2])]).shape


# In[27]:


models['model_d'].predict(x_train[:3])


# In[28]:


zsamples = np.random.normal(size=(10 * 10, latent_dim))
def generator_sampler(latent_dim, generator, zsamples=zsamples):
    def fun(zsamples=zsamples):
        #zsamples = np.random.normal(size=(10 * 10, latent_dim))
        gen = generator.predict([zsamples, np.repeat(np.arange(10).reshape(1,-1), 10, axis=0).flatten()])
        return gen.reshape((10, 10, 28, 28))
    return fun


# In[29]:


from keras.callbacks import Callback
import os
from matplotlib import pyplot as plt, gridspec


def write_image_grid(filepath, imgs, figsize=None, cmap='gray'):
    directory = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig = create_image_grid(imgs, figsize, cmap=cmap)
    fig.savefig(filepath)
    plt.close(fig)


def create_image_grid(imgs, figsize=None, cmap='gray'):
    n = imgs.shape[0]
    m = imgs.shape[1]
    if figsize is None:
        figsize = (n, m)
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(n, m)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.
    for i in range(n):
        for j in range(m):
            ax = plt.subplot(gs1[i, j])
            img = imgs[i, j, :]
            ax.imshow(img, cmap=cmap)
            ax.axis('off')
    return fig


class ImageGridCallback(Callback):
    def __init__(self, image_path, generator, cmap='gray'):
        self.image_path = image_path
        self.generator = generator
        self.cmap = cmap

    def on_epoch_end(self, epoch, logs={}):
        xsamples = self.generator()
        image_path = self.image_path.format(epoch)
        write_image_grid(image_path, xsamples, cmap=self.cmap)


# In[30]:


# train model
generator_cb = ImageGridCallback("output/gan_convolutional/epoch-{:03d}.png",
                                 generator_sampler(latent_dim, generator))
callbacks = [generator_cb]
if K.backend() == "tensorflow":
    callbacks.append(
        TensorBoard(log_dir=os.path.join("output/gan_convolutional/", "logs/"),
                    histogram_freq=0, write_graph=True, write_images=True))


# In[31]:


# loss_min_dic = {
#     'd_valid': 0.15,
#     'd_fake': 0.15,
#     'gd_valid': 0.5,
# }
loss_min_dic = {
    'd_valid': 0.0,
    'd_fake': 0.0,
    'gd_valid': 0.0,
}


# In[35]:


class Train(object):
    
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y, batch_size=32, epochs=5, callbacks=[], verbose=0):
        
        for ii in range(epochs):
            idx = np.random.permutation(np.arange(X.shape[0]))
            it = range(0, X.shape[0], batch_size)
            d_loss_valid = np.inf
            d_loss_fake = np.inf
            g_loss = np.inf
            d_loss_valid_list = []
            d_loss_fake_list = []
            g_loss_list = []
            
            with tqdm(total=len(it), file=sys.stdout,
                      disable=False if 1<verbose else True) as pbar:
                for jj in it:
                    idx_selected = idx[jj:(jj+batch_size) if (jj+batch_size)<X.shape[0] else X.shape[0]]
                    valid = np.ones((len(idx_selected), ))
                    fake = np.zeros((len(idx_selected), ))
                    
                    imgs = X[idx_selected]
                    labels = y[idx_selected]
                    labels_cats = to_categorical(labels, num_classes=10)
                    
                    rlabels = np.random.randint(0, 10, (len(idx_selected),))
                    rlabels_cats = to_categorical(rlabels, num_classes=10)
                    
                    z = np.random.normal(0, 1, (len(idx_selected), latent_dim))
                    imgs_gen = self.models['model_g'].predict([z, rlabels])
                    
#                     if len(d_loss_valid_list) == 0 or \
#                        loss_min_dic['d_valid'] < np.array(d_loss_valid_list).mean():
#                         d_loss_valid = self.models['model_d'].train_on_batch(imgs, valid)
#                     else:
#                         d_loss_valid = self.models['model_d'].evaluate(imgs, valid, verbose=0)
#                     d_loss_valid_list.append(d_loss_valid)
                    
#                     if len(d_loss_fake_list) == 0 or \
#                        loss_min_dic['d_fake'] < np.array(d_loss_fake_list).mean():
#                         d_loss_fake = self.models['model_d'].train_on_batch(imgs_gen, fake)
#                     else:
#                         d_loss_fake = self.models['model_d'].evaluate(imgs_gen, fake, verbose=0)
#                     d_loss_fake_list.append(d_loss_fake)
                    if len(d_loss_valid_list)==0 or len(d_loss_fake_list)==0:
                        '''init'''
                        d_loss_valid = self.models['model_d'].train_on_batch(imgs, [valid, labels_cats])
                        d_loss_fake = self.models['model_d'].train_on_batch(imgs_gen, [fake, rlabels_cats])
                    elif np.array(d_loss_fake_list).mean() < loss_min_dic['d_fake']:
                        '''d_fake : no train'''
                        d_loss_valid = self.models['model_d'].train_on_batch(imgs, [valid, labels_cats])
                        d_loss_fake = self.models['model_d'].evaluate(imgs_gen, [fake, rlabels_cats], verbose=0)
                    elif np.array(d_loss_valid_list).mean() < loss_min_dic['d_valid']:
                        '''d_valid : no train'''
                        d_loss_valid = self.models['model_d'].evaluate(imgs, [valid, labels_cats], verbose=0)
                        d_loss_fake = self.models['model_d'].train_on_batch(imgs_gen, [fake, rlabels_cats])
                    else:
                        d_loss_valid = self.models['model_d'].train_on_batch(imgs, [valid, labels_cats])
                        d_loss_fake = self.models['model_d'].train_on_batch(imgs_gen, [fake, rlabels_cats])
                    d_loss_valid_list.append(d_loss_valid[0])
                    d_loss_fake_list.append(d_loss_fake[0])
                    
                    if len(g_loss_list) == 0 or                        loss_min_dic['gd_valid'] < np.array(g_loss_list).mean():
                        g_loss = self.models['model_gd'].train_on_batch([z, rlabels], [valid, rlabels_cats])
                        #g_loss = self.models['model_gd'].evaluate([z, rlabels], [valid, rlabels_cats], verbose=0)
                    else:
                        g_loss = self.models['model_gd'].evaluate([z, rlabels], [valid, rlabels_cats], verbose=0)
                    g_loss_list.append(g_loss[0])
                    
                    if 2<verbose:
                        pbar.set_description("epoch: %d [d_valid_loss: %f(%f) d_fake_loss: %f(%f) gd_loss: %f(%f)]" % 
                                             (ii, np.array(d_loss_valid_list).mean(), d_loss_valid[0],
                                                  np.array(d_loss_fake_list).mean(), d_loss_fake[0], 
                                                  np.array(g_loss_list).mean(), g_loss[0]))
                    pbar.update(1)
            
            #break
            if 0<verbose:
                print("epoch: %d [d_valid_loss: %f(%f) d_fake_loss: %f(%f) gd_loss: %f(%f)]" % 
                                     (ii, np.array(d_loss_valid_list).mean(), d_loss_valid[0],
                                          np.array(d_loss_fake_list).mean(), d_loss_fake[0],
                                          np.array(g_loss_list).mean(), g_loss[0]))
            if (np.array(g_loss_list).mean() < 0.000001) or                (d_loss_valid[0] < 0.000001 and g_loss[0] < 0.000001):
                print('create generator !!')
                generator = model_generator()
                print('create discriminator !!')
                discriminator = model_discriminator(input_shape=input_shape)
                self.models = make_models(generator, discriminator)
            generator_cb.on_epoch_end(ii)


# In[37]:


train = Train(models)
train.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1)
models = train.models


# In[ ]:





# In[ ]:


models['model_d'].predict(x_train[:3])


# In[ ]:


models['model_g'].predict([zsamples, np.repeat(np.arange(10).reshape(1,-1), 10, axis=0).flatten()]).shape


# In[ ]:


plt.imshow(models['model_g'].predict([zsamples, np.repeat(np.arange(10).reshape(1,-1), 10, axis=0).flatten()])[0].reshape((28,28)))


# In[ ]:





# In[ ]:


ls -la output/gan_convolutional


# In[ ]:


res = generator.predict([zsamples, np.repeat(np.arange(10).reshape(1,-1), 10, axis=0).flatten()])
res.shape


# In[ ]:


plt.imshow(res[9].reshape((28,28)))


# In[ ]:


x_train[0].shape


# In[ ]:


discriminator.predict(x_train[:5]), y_train[:5]


# In[ ]:




