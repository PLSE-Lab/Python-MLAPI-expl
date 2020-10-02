#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, LeakyReLU, Reshape
from keras.optimizers import Adam
from numpy.random import randint, rand, randn
from numpy import ones, zeros
from matplotlib import pyplot
import numpy


# In[ ]:


def load_data(path):
    with numpy.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# In[ ]:


def load_real_data():
    (trainX, _), (_, _) = load_data('../input/mnist-numpy/mnist.npz')
    X = numpy.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    return X

def gen_real_samples(data, n_samples):
    ix = randint(0, data.shape[0], n_samples)
    
    X = data[ix]
    
    y = numpy.ones((n_samples, 1))
    return X, y


# In[ ]:


def define_discriminator(in_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(lr=0.002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# In[ ]:


def define_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model


# In[ ]:


def gen_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# In[ ]:


def gen_fake_samples(g_model, latent_dim, n_samples):
    x_input = gen_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y


# In[ ]:


def define_gan(g, d):
    d.trainable = False
    model = Sequential()
    model.add(g)
    model.add(d)
    adam = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    return model


# In[ ]:


def train(g, d, gan, data, latent_dim, n_epochs=1000, batch=256):
    bat_per_epo = int(data.shape[0] / batch)
    half_batch = int(batch / 2)
    
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = gen_real_samples(data, half_batch)
            X_fake, y_fake = gen_fake_samples(g, latent_dim, half_batch)
            
            X, y = numpy.vstack((X_real, X_fake)), numpy.vstack((y_real, y_fake))
            
            d_loss, _ = d.train_on_batch(X, y)
            
            X_gan = gen_latent_points(latent_dim, batch)
            y_gan = ones((batch, 1))
            
            g_loss = gan.train_on_batch(X_gan, y_gan)
            
            print("Epoch: %d/%d, Batch: %d/%d" % (i,n_epochs, j,bat_per_epo), d_loss, " | ", g_loss)
        if d_loss > 0.0 and d_loss < 1.0:
            print("SAVING FILE")
            g.save("gen_model_%03d_%.02f%%.h5" % (i + 1, d_loss))


# In[ ]:


g = define_generator(25)
d = define_discriminator()
gan = define_gan(g, d)


# In[ ]:


train(g, d, gan, load_real_data(), 25)


# In[ ]:


# Generatred MNIST
X = g.predict(gen_latent_points(25, 16))

for i in range(X.shape[0]):
    pyplot.subplot(4,4,1+i)
    pyplot.imshow(X[i].reshape(28,28))

pyplot.show()

