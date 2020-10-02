#!/usr/bin/env python
# coding: utf-8

# # GAN from scratch

# ![gan_architecture-1.png](attachment:gan_architecture-1.png)

# The structure of the code and Image designed by Christina Kouridi<br>
# https://christinakouridi.blog

# ## Content table
# 1. [Definition](#definition)<br>
# 2. [Exploratory Analysis](#exploratory)<br>
# 3. [Implementation](#implementation)<br>
#     3.1 [Activation Functions](#activation)<br>
#     3.2 [Plot Function](#plot)<br>
#     3.3 [Forward Functions](#forward)<br>
#     3.4 [Backward Functions](#backward)<br>
#     3.5 [Cross-Entropy Loss Function](#loss)<br>
#     3.3 [Training](#training)<br>

# <a id='definition'></a>
# # Definition

# A generative adversarial network (GAN) is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. Two neural networks contest with each other in a game (in the sense of game theory, often but not always in the form of a zero-sum game). Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics. Though originally proposed as a form of generative model for unsupervised learning, GANs have also proven useful for semi-supervised learning, fully supervised learning, and reinforcement learning.
# 
# https://en.wikipedia.org/wiki/Generative_adversarial_network

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.datasets import mnist


# <a id='exploratory'></a>
# # Exploratory Data Analysis

# In[ ]:


df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
df.head()


# In[ ]:


categories = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
idx_category = {k: v for v, k in enumerate(categories)}


# In[ ]:


plt.figure(figsize=(14, 12))

for i in range(0,20):
    splt = plt.subplot(7, 10, i+1)
    plt.imshow(df.iloc[:, 1:].values[i].reshape(28, 28))
    plt.title("{}".format(categories[df.iloc[:, 0].values[i]]))
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()


# In[ ]:


labels = df.iloc[:, 0].to_numpy()
train = df.iloc[:, 1:].to_numpy()


# In[ ]:


train = train.reshape(train.shape[0], 28, 28)


# <a id='implementation'></a>
# # Implementation

# ### Setup parameters

# In[ ]:


params = {}
params['epochs'] = 100
params['batch_size'] = 64
params['nx_g'] = 100
params['nh_g'] = 128
params['nh_d'] = 128
params['lr'] = 1e-3
params['dr'] = 1e-4
params['image_size'] = 28
params['display_epochs'] = 5


# In[ ]:


theta = {}
gamma = {}

# Generator
theta['W0_g'] = np.random.randn(params['nx_g'], params['nh_g']) * np.sqrt(2. / params['nx_g'])  # 100x128
theta['b0_g'] = np.zeros((1, params['nh_g']))  # 1x100

theta['W1_g'] = np.random.randn(params['nh_g'], params['image_size'] ** 2) * np.sqrt(2. / params['nh_g'])  # 128x784
theta['b1_g'] = np.zeros((1, params['image_size'] ** 2))  # 1x784

# Discriminator
theta['W0_d'] = np.random.randn(params['image_size'] ** 2, params['nh_d']) * np.sqrt(2. / params['image_size'] ** 2)  # 784x128
theta['b0_d'] = np.zeros((1, params['nh_d']))  # 1x128

theta['W1_d'] = np.random.randn(params['nh_d'], 1) * np.sqrt(2. / params['nh_d'])  # 128x1
theta['b1_d'] = np.zeros((1, 1))  # 1x1


# <a id='activation'></a>
# ### Activation functions and derivatives

# In[ ]:


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dsigmoid(x):
    y = sigmoid(x)
    return y * (1. - y)


def dtanh(x):
    return 1. - np.tanh(x) ** 2


def lrelu(x, alpha=1e-2):
    return np.maximum(x, x * alpha)


def dlrelu(x, alpha=1e-2):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


# <a id='plot'></a>
# ### Plot image function

# In[ ]:


def sample_images(images, epoch, show):
        images = np.reshape(images, (params['batch_size'], params['image_size'], params['image_size']))

        fig = plt.figure(figsize=(4, 4))

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        if show == True:
            plt.show()
        else:
            plt.close()


# <a id='forward'></a>
# ### Forward functions

# In[ ]:


def forward_generator(z):
        gamma['z0_g'] = np.dot(z, theta['W0_g']) + theta['b0_g']
        gamma['a0_g'] = lrelu(gamma['z0_g'], alpha=0)

        gamma['z1_g'] = np.dot(gamma['a0_g'], theta['W1_g']) + theta['b1_g']
        gamma['a1_g'] = np.tanh(gamma['z1_g'])
     
    
def forward_discriminator(x):
        gamma['z0_d'] = np.dot(x, theta['W0_d']) + theta['b0_d']
        gamma['a0_d'] = lrelu(gamma['z0_d'])

        z1_d = np.dot(gamma['a0_d'], theta['W1_d']) + theta['b1_d']
        a1_d = sigmoid(z1_d)
        return z1_d, a1_d


# ### Complementary functions

# In[ ]:


def update(theta, grads, lr):
    for i,t in enumerate(theta):
        theta[i] -= grads[i] * lr


# In[ ]:


def combine_real_fake_grads(real_grads, fake_grads):
    grads = np.array(real_grads) + np.array(fake_grads)
    return grads


# <a id="backward"/></a>
# ### Backward functions

# In[ ]:


def backward_discriminator(x_real, z1_real, a1_real, x_fake, z1_fake, a1_fake):
        da1_real = -1. / (a1_real + 1e-8)  # 64x1

        dz1_real = da1_real * dsigmoid(z1_real)  # 64x1
        db1_real = np.sum(dz1_real, axis=0, keepdims=True)
        dW1_real = np.dot(gamma['a0_d'].T, dz1_real)

        da0_real = np.dot(dz1_real, theta['W1_d'].T)
        dz0_real = da0_real * dlrelu(gamma['z0_d'])
        
        db0_real = np.sum(dz0_real, axis=0, keepdims=True)
        dW0_real = np.dot(x_real.T, dz0_real)

        # fake input gradients -np.log(1 - a1_fake)
        da1_fake = 1. / (1. - a1_fake + 1e-8)

        dz1_fake = da1_fake * dsigmoid(z1_fake)
        db1_fake = np.sum(dz1_fake, axis=0, keepdims=True)
        dW1_fake = np.dot(gamma['a0_d'].T, dz1_fake)

        da0_fake = np.dot(dz1_fake, theta['W1_d'].T)
        dz0_fake = da0_fake * dlrelu(gamma['z0_d'], alpha=0)
        
        db0_fake = np.sum(dz0_fake, axis=0, keepdims=True)
        dW0_fake = np.dot(x_fake.T, dz0_fake)

        # Combine gradients for real & fake images
        grads = combine_real_fake_grads(np.array([dW0_real, dW1_real, db0_real, db1_real]), np.array([dW0_fake, dW1_fake, db0_fake, db1_fake]))

        # Update gradients
        update([theta['W0_d'], theta['W1_d'], theta['b0_d'], theta['b1_d']], grads, params['lr'])


# In[ ]:


def backward_generator(z, x_fake, z1_fake, a1_fake):
        da1_d = -1.0 / (a1_fake + 1e-8)  # 64x1

        dz1_d = da1_d * dsigmoid(z1_fake)
        da0_d = np.dot(dz1_d, theta['W1_d'].T)
        dz0_d = da0_d * dlrelu(gamma['z0_d'])
        dx_d = np.dot(dz0_d, theta['W0_d'].T)

        # Backprop through Generator
        dz1_g = dx_d * dtanh(gamma['z1_g'])
        dW1_g = np.dot(gamma['a0_g'].T, dz1_g)
        db1_g = np.sum(dz1_g, axis=0, keepdims=True)

        da0_g = np.dot(dz1_g, theta['W1_g'].T)
        dz0_g = da0_g * dlrelu(gamma['z0_g'], alpha=0)
        dW0_g = np.dot(z.T, dz0_g)
        db0_g = np.sum(dz0_g, axis=0, keepdims=True)

        # Update gradients
        update([theta['W0_g'], theta['W1_g'], theta['b0_g'], theta['b1_g']], [dW0_g, dW1_g, db0_g, db1_g], params['lr'])


# ### Split data by batches and label

# In[ ]:


def preprocess_data(x, y):
        x_train = []
        y_train = []

        # limit the data to a subset of digits from 0-9
        for i in range(y.shape[0]):
            if y[i] in params['numbers']:
                x_train.append(x[i])
                y_train.append(y[i])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # limit the data to full batches only
        num_batches = x_train.shape[0] // params['batch_size']
        x_train = x_train[: num_batches * params['batch_size']]
        y_train = y_train[: num_batches * params['batch_size']]

        # flatten the images (_,28,28)->(_, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))

        # normalise the data to the range [-1,1]
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        # shuffle the data
        idx = np.random.permutation(len(x_train))
        x_train, y_train = x_train[idx], y_train[idx]
        return x_train, y_train, num_batches


# In[ ]:


discim_losses = []  # stores the disciminator losses
gener_losses = []  # stores the generator losses


# <a id='loss'></a>
# ### Cross-Entropy Loss Function

# In[ ]:


def loss_fn(a1_d_real, a1_d_fake):
    # Cross Entropy
    discim_loss = np.mean(-np.log(a1_d_real) - np.log(1 - a1_d_fake))
    discim_losses.append(discim_loss)

    gener_loss = np.mean(-np.log(a1_d_fake))
    gener_losses.append(gener_loss)
    
    return gener_loss, discim_loss


# <a id='training'></a>
# ### Training

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.33, random_state=42)


# In[ ]:


# Chose the item to GAN
params['numbers'] = [idx_category['t-shirt']]


# In[ ]:


# preprocess input; note that labels aren't needed
x_train, _, num_batches = preprocess_data(X_train, y_train)

for epoch in range(params['epochs']):
    for i in range(num_batches):
        # Prepare unput and z - noise
        x_real = x_train[i * params['batch_size']: (i + 1) * params['batch_size']]
        z = np.random.normal(0, 1, size=[params['batch_size'], params['nx_g']])  # 64x100

        # Forward
        forward_generator(z)

        z1_d_real, a1_d_real = forward_discriminator(x_real)
        z1_d_fake, a1_d_fake = forward_discriminator(gamma['a1_g'])

        # Cross Entropy
        gener_loss, discim_loss = loss_fn(a1_d_real, a1_d_fake)
        
        # Backward
        backward_discriminator(x_real, z1_d_real, a1_d_real, gamma['a1_g'], z1_d_fake, a1_d_fake)
        backward_generator(z, gamma['a1_g'], z1_d_fake, a1_d_fake)

    if epoch % params['display_epochs'] == 0:
        print("Epoch : ", epoch, " Loss: ", gener_loss)
        sample_images(gamma['a1_g'], epoch, show=True)

    # reduce learning rate after every epoch
    params['lr'] = params['lr'] * (1.0 / (1.0 + params['dr'] * epoch))

