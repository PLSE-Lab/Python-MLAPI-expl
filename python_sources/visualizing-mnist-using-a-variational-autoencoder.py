#!/usr/bin/env python
# coding: utf-8

# # Visualizing MNIST with a Deep Variational Autoencoder
# 
# 1. Introduction  
#     A. What is autoencoding?  
#     B. Autoencoders  
#     C. The Variational Variety  
# 2. Data preparation  
#     A. Load data  
#     B. Combine train & test  
#     C. Split into new training & validation sets  
#     D. Reshape & normalize  
# 3. Model construction  
#     A. Encoder network  
#     B. Sampling function  
#     C. Decoder network  
#     D. Loss
# 4. Train the VAE
# 5. Results  
#     A. Clustering of digits in the latent space  
#     B. Sample digits

# ## *1. Introduction*
# I spent a rainy weekend recently learning about autoencoders using TensorFlow and Keras in Python. I learned that variational autoencoders (VAEs) can be used to visualize high-dimensional data in a meaningful, lower-dimensional space. In this kernel, I go over some details about autoencoding and autoencoders, especially VAEs, before constructing and training a deep VAE on the MNIST data from the Digit Recognizer competition. We'll see how the data cluster in the lower-dimensional space according to their digit class. Plotting the test set data in this space shows where the images with unknown digit classes fall with respect to the known digit classes.
# 
# The code here borrows heavily from Fran&ccedil;ois Chollet's example VAE from his book [Deep Learning with Python](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=sr_1_1?ie=UTF8&qid=1520470984&sr=8-1&keywords=francois+chollet). You can find a repo of examples from the book (including the one that inspired this kernel) [here on GitHub](https://github.com/brilliantFire/deep-learning-with-python-notebooks).
# 
# ### *A. What is autoencoding?*
# Autoencoding is much like what it sounds in the sense that the input and 'output' are essentially the same. It's an algorithm for data compression where the functions for compression and decompression are *learned from the data*. It's considered more of a *semi-supervised* learning method as opposed to a truly *unsupervised* one since it's not entirely 'targetless'. Instead it learns the targets from the data itself.
# 
# Despite all this talk of data compression, autoencoders aren't typically used for that purpose. In practice, you're much more likely to see them being used to preprocess data (as in denoising - think images but it doesn't have to be ;) ) or for dimensionality reduction. In fact, the hidden layers of simple autoencoders are doing something like principal component analysis (PCA), another method traditionally used for dimensionality reduction.
# 
# ### *B. Autoencoders*
# Generally autoencoders have three parts: an encoder, a decoder, and a 'loss' function that maps one to the other. For the simplest autoencoders - the sort that compress and then reconstruct the original inputs from the compressed representation - we can think of the 'loss' as describing the amount of information lost in the process of reconstruction. Typically when people are talking about autoencoders, they're talking about ones where the encoders and decoders are neural networks (in our case deep convnets). In training the autoencoder, we're optimizing the parameters of the neural networks to minimize the 'loss' (or distance) and we do that by stochastic gradient descent (yet another topic for another post). 
# 
# ### *C. The Variational Variety*
# There's a bunch of different kinds of autoencoders but for this post I'm going to concentrate on one type called a *variational autoencoder*. Variational autoencoders (VAEs) don't learn to morph the data in and out of a compressed representation of itself like the 'vanilla' autoencoders I described above. Instead, they learn the parameters of the probability distribution that the data came from. These types of autoencoders have much in common with latent factor analysis (if you know something about that). The encoder and decoder learn models that are in terms of underlying, unobserved *latent* variables. It's essentially an inference model and a generative model daisy-chained together.
# 
# ![A variational autoencoder](https://i.imgur.com/ZN6MyTx.png)
# 
# VAEs have received a lot of attention because of their *generative* ability (though they seem to be falling out of fashion in favor of general adversarial networks, or GANs, in that regard). Since they learn about the distribution the inputs came from, we can sample from that distribution to generate novel data. As we'll see, VAEs can also be used to cluster data in useful ways.

# ## 2. *Data preparation*  
# ### *A. Load Data*

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import norm

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

K.clear_session()

np.random.seed(237)


# In[ ]:


train_orig = pd.read_csv('../input/train.csv')
test_orig = pd.read_csv('../input/test.csv')

train_orig.head()


# ### *B. Combine Train & Test*
# Let's add a placeholder 'label' column to the test dataset. We'll give all the test images a label of '11' for now ("This data goes to 11.") 

# In[ ]:


# create 'label' column in test dataset; rearrange so that columns are in the same order as in train
test_orig['label'] = 11
testCols = test_orig.columns.tolist()
testCols = testCols[-1:] + testCols[:-1]
test_orig = test_orig[testCols]


# We want to train the autoencoder with as many images as possible. Also, since we don't need the labels for building the model (remember: *semi-supervised*), it makes sense to combine the train and test data into one combined dataframe.

# In[ ]:


# combine original train and test sets
combined = pd.concat([train_orig, test_orig], ignore_index = True)

combined.head()


# In[ ]:


combined.tail()


# ### *C. Split into training & validation sets*
# Despite being trained in a semi-supervised way, the VAE algorithm entails minimizing a 'loss' function. As we'll see shortly, that loss is actually two different 'losses' combined, one that describes the difference between the input images and the images reconstructed from samples from the latent distribution, and another that is the difference between the latent distribution and the prior (the inputs). We can calculate this loss on a validation set with each training epoch as an estimate of how the model describes data it was not trained on.  
# 
# Remember that we've combined the original train and test data so the *new* train and validation sets that we make below will each have some images with missing ('11') labels.

# In[ ]:


# Hold out 5000 random images as a validation/test sample
valid = combined.sample(n = 5000, random_state = 555)
train = combined.loc[~combined.index.isin(valid.index)]

# free up some space and delete test and combined
del train_orig, test_orig, combined

valid.head()


# ### *4. Reshape & normalize*
# Our encoder and decoder are deep convnets constructed using the Keras Functional API. We'll need to separate the inputs from the labels, normalize them by dividing the max pixel value, and reshape them into 28x28 pixel images.

# In[ ]:


# X's
X_train = train.drop(['label'], axis = 1)
X_valid = valid.drop(['label'], axis = 1)

# labels
y_train = train['label']
y_valid = valid['label']

# Normalize and reshape
X_train = X_train.astype('float32') / 255.
X_train = X_train.values.reshape(-1,28,28,1)

X_valid = X_valid.astype('float32') / 255.
X_valid = X_valid.values.reshape(-1,28,28,1)


# We can take a look at a few random images. The bottom right panel shows one of the more difficult-to-classify digits (even for humans!).

# In[ ]:


plt.figure(1)
plt.subplot(221)
plt.imshow(X_train[13][:,:,0])

plt.subplot(222)
plt.imshow(X_train[690][:,:,0])

plt.subplot(223)
plt.imshow(X_train[2375][:,:,0])

plt.subplot(224)
plt.imshow(X_train[42013][:,:,0])
plt.show()


# ## 3. *Model construction*
# ### *A. Encoder network*
# A VAE has three basic parts:  
# 
# 1. An encoder that that learns the parameters (mean and variance) of the underlying latent distribution;  
# 2. A means of sampling from that distribution; and,  
# 3. A decoder that can turn the sample from #2 back into an image.  
# 
# In this example, both the encoder and decoder networks are deep convnets. You'll notice that the encoder below has two output layers, one for the latent distribution mean (z_mu) and the other for its variance (z_log_sigma).

# In[ ]:


img_shape = (28, 28, 1)    # for MNIST
batch_size = 16
latent_dim = 2  # Number of latent dimension parameters

# Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', 
                  activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
# need to know the shape of the network here for the decoder
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# Two outputs, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)


# ### *B. Sampling function*
# Next, we create a function to sample from the distribution we just learned the parameters of. `epsilon` is a tensor of small random normal values. One of the assumptions underlying a VAE like this is that our data arose from a random process and is normally distributed in the latent space.
# 
# With Keras, everything has to be in a 'layer' to compile correctly. This goes for our sampling function. The `Lambda` layer wrapper let's us do this.

# In[ ]:


# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon

# sample vector from the latent distribution
z = layers.Lambda(sampling)([z_mu, z_log_sigma])


# ### *C. Decoder network*
# The decoder is basically the encoder in reverse.

# In[ ]:


# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:])

# Expand to 784 total pixels
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# reshape
x = layers.Reshape(shape_before_flattening[1:])(x)

# use Conv2DTranspose to reverse the conv layers from the encoder
x = layers.Conv2DTranspose(32, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)
x = layers.Conv2D(1, 3,
                  padding='same', 
                  activation='sigmoid')(x)

# decoder model statement
decoder = Model(decoder_input, x)

# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)


# ### *D. Loss*
# We need one more thing and that's something that will calculate the unique loss function the VAE requires. Recall that the VAE is trained using a loss function with two components:  
# 
# 1. 'Reconstruction loss' - This is the cross-entropy describing the errors between the decoded samples from the latent distribution and the original inputs.  
# 2. The Kullback-Liebler divergence between the latent distribution and the prior (this acts as a sort of regularization term).  
# 
# We define a custom layer class that calculates the loss. 

# In[ ]:


# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomVariationalLayer()([input_img, z_decoded])


# Now we can instantiate the model and take a look at its summary.

# In[ ]:


# VAE model statement
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


# ## 4. *Train the VAE*
# Finally, we fit the model.

# In[ ]:


vae.fit(x=X_train, y=None,
        shuffle=True,
        epochs=7,
        batch_size=batch_size,
        validation_data=(X_valid, None))


# ## 5. *Results*  
# ### *A. Clustering of digits in the latent space*
# We can make predictions on the validation set using the encoder network. This has the effect of translating the images from the 784-dimensional input space into the 2-dimensional latent space. When we color-code those translated data points according to their *known* digit class, we can see how the digits cluster together.

# In[ ]:


# Isolate original training set records in validation set
valid_noTest = valid[valid['label'] != 11]

# X's and Y's
X_valid_noTest = valid_noTest.drop('label', axis=1)
y_valid_noTest = valid_noTest['label']

# Reshape and normalize
X_valid_noTest = X_valid_noTest.astype('float32') / 255.
X_valid_noTest = X_valid_noTest.values.reshape(-1,28,28,1)


# In[ ]:


# Translate into the latent space
encoder = Model(input_img, z_mu)
x_valid_noTest_encoded = encoder.predict(X_valid_noTest, batch_size=batch_size)
plt.figure(figsize=(10, 10))
plt.scatter(x_valid_noTest_encoded[:, 0], x_valid_noTest_encoded[:, 1], c=y_valid_noTest, cmap='brg')
plt.colorbar()
plt.show()


# Including the original test set data lets us see where they fall with respect to the known digit clusters.

# In[ ]:


# set colormap so that 11's are gray
custom_cmap = matplotlib.cm.get_cmap('brg')
custom_cmap.set_over('gray')

x_valid_encoded = encoder.predict(X_valid, batch_size=batch_size)
plt.figure(figsize=(10, 10))
gray_marker = mpatches.Circle(4,radius=0.1,color='gray', label='Test')
plt.legend(handles=[gray_marker], loc = 'best')
plt.scatter(x_valid_encoded[:, 0], x_valid_encoded[:, 1], c=y_valid, cmap=custom_cmap)
plt.clim(0, 9)
plt.colorbar()


# ### *B. Sample digits*
# Another fun thing we can do is to use the decoder network to take a peak at what samples from the latent space look like as we change the latent variables. What we end up with is a smoothly varying space where each digit transforms into the others as we dial the latent variables up and down.

# In[ ]:


# Display a 2D manifold of the digits
n = 20  # figure with 20x20 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Construct grid of latent variable values
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# decode for each square in the grid
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')
plt.show()  


# Thanks for reading!
