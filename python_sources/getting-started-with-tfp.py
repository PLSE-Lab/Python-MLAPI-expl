#!/usr/bin/env python
# coding: utf-8

# This notebook helps you to get started with Tensorflow Probability, a module for probabilistic reasoning within the DL framework.  
# Why do we want to think probabilistically?  The simplest answer is that we want to have the ability to make prediction with uncertainty estimation. This would help us understand the data better and to make decision based on probabilistic reasoning.  
# Another application for Tensorflow Probability is in generative model. In models such as GANs or Variational Autoencoder, we need the probability module to have random initialization.  
# In this notebook, we will start with an autoencoder. Next, we will build a variational autoencoder that take advantages of Tensorflow Probability model. Lastly, we will make a probablistic regression model.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().system('pip install tensorflow==2.0.0-alpha0')
get_ipython().system('pip install tensorflow_datasets')
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import make_regression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data_train.head()
data_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


x_train = data_train.drop(['label'], axis = 1).values / 255
y_train = data_train['label'].values
x_test = data_test.values / 255


# Now we implement an autoencoder. It is a model for feature extraction. The latent space contains the compressed representation of the original data.

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(392, activation = 'relu'),
    tf.keras.layers.Dense(196, activation = 'relu'),
    tf.keras.layers.Dense(392, activation = 'relu'),
    tf.keras.layers.Dense(784, activation = 'sigmoid')
])
model.compile(optimizer = 'adam', loss = 'mse')


# In[ ]:


model.fit(x_train, x_train, epochs = 5, verbose = 0)


# In[ ]:


ix = np.random.randint(0, len(x_test), 2)
test = x_test[ix]
test_outcome = model.predict(test)
f, ax = plt.subplots(2, 2, figsize = (5, 5))
ax[0, 0].imshow(test[0].reshape(28, 28))
ax[0, 0].set_title('original')
ax[1, 0].imshow(test[1].reshape(28, 28))
ax[0, 1].imshow(test_outcome[0].reshape(28, 28))
ax[0, 1].set_title('reconstructed')
ax[1, 1].imshow(test_outcome[1].reshape(28, 28))


# We now have an autoencoder that knows how to reconstruct hand-written digits, even though it have not seen the digit before. However, the reconstructed image will always look the same as the original, making it useless as a generative model.  
# Next, we will try to understand the core concept of variational autoencoder by writing one.

# In[ ]:


encoded_size = 16
prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = (784, )),
    tf.keras.layers.Dense(392, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(196, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(encoded_size),
               activation=None),
    tfp.layers.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior)),
])
encoder.summary()


# In[ ]:


decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = [encoded_size]),
    tf.keras.layers.Dense(392, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(784, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tfp.layers.IndependentBernoulli((784, ), tfp.distributions.Bernoulli.logits),
])
decoder.summary()


# In[ ]:


vae = tf.keras.Model(inputs = encoder.inputs, outputs = decoder(encoder.outputs[0]))
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
vae.compile(optimizer = tf.optimizers.Adam(learning_rate=1e-4),
            loss = negloglik)
vae.fit(x_train, x_train, epochs = 20, verbose = 0)
vae.summary()


# In[ ]:


ix = np.random.randint(0, len(x_test), 2)
test = x_test[ix]
test_outcome = vae(test)
f, ax = plt.subplots(2, 3, figsize = (10, 5))
ax[0, 0].imshow(test[0].reshape(28, 28))
ax[0, 0].set_title('original')
ax[1, 0].imshow(test[1].reshape(28, 28))

ax[0, 1].imshow(np.array(test_outcome.mode()[0]).reshape(28, 28))
ax[0, 1].set_title('reconstructed mode')
ax[1, 1].imshow(np.array(test_outcome.mode()[1]).reshape(28, 28))

ax[0, 2].imshow(np.array(test_outcome.mean()[0]).reshape(28, 28))
ax[0, 2].set_title('reconstructed mean')
ax[1, 2].imshow(np.array(test_outcome.mean()[1]).reshape(28, 28))


# Now, we use priors to generate never-seen-before digits:

# In[ ]:


z = prior.sample(10)
xtilde = decoder(z)
f, ax = plt.subplots(1, 10, figsize = (20, 10))
for i in range(10):
    ax[i].imshow(np.array(xtilde.sample()[i]).reshape(28, 28))
    ax[i].axis('off')
plt.show()

f, ax = plt.subplots(1, 10, figsize = (20, 10))
for i in range(10):
    ax[i].imshow(np.array(xtilde.mode()[i]).reshape(28, 28))
    ax[i].axis('off')
plt.show()

f, ax = plt.subplots(1, 10, figsize = (20, 10))
for i in range(10):
    ax[i].imshow(np.array(xtilde.mean()[i]).reshape(28, 28))
    ax[i].axis('off')
plt.show()


# Bravo!  
# Since we now know how tensorflow probability works, let's jump into a toy dataset and see it predicting 95% confidence interval in a regression problem.

# In[ ]:


x = np.arange(0, 5, 0.01)
y = 1 - np.exp(- x) + np.random.rand(x.shape[0]) * np.sqrt(x)
plt.scatter(x, y)


# In[ ]:


r_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation = None)
])
r_model.compile(optimizer = 'adam', loss = 'mse')


# In[ ]:


r_model.fit(x.reshape(-1, 1), y.reshape(-1, 1), epochs = 100, verbose = 0)


# In[ ]:


y_pred = r_model.predict(x.reshape(-1, 1))
plt.scatter(x, y)
plt.scatter(x, y_pred.reshape(-1))


# In[ ]:


r_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2,),
    tfp.layers.DistributionLambda(
      lambda t: tfp.distributions.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])))
])


# In[ ]:


negloglik = lambda x, rv_x: -rv_x.log_prob(x)
r_model.compile(optimizer = tf.optimizers.Adam(learning_rate=0.05), loss = negloglik)
r_model.fit(x.reshape(-1, 1), y.reshape(-1, 1), epochs = 100, verbose = 0)
r_model.summary()


# In[ ]:


yhat = r_model(x.reshape(-1, 1))
mean = yhat.mean()
stddev = yhat.stddev()
mean_plus_stddev = mean - 1.96 * stddev
mean_minus_stddev = mean + 1.96 * stddev

plt.scatter(x, y)
plt.plot(x, np.array(mean).reshape(-1))
plt.plot(x, np.array(mean_plus_stddev).reshape(-1))
plt.plot(x, np.array(mean_minus_stddev).reshape(-1))

