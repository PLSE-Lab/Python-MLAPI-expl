#!/usr/bin/env python
# coding: utf-8

# # GENERALIZED IMAGE GENERATOR
# Generative Adversarial Networks (GAN) are useful in many ways: Learning from existing samples to create your own nearly-real sample. 
# It is uniquely different from already existing alternatives to create more data. For example, data augmentation works for only specified criteria. We may need to randomize the image creation process, so GANs are the most brilliant method yet.
# 
# The code below generalizes the whole process of running a GAN model: You just need to initialize the variables below and make sure the images have consistent weight, height, and channels. In case your data is inconsistent, you need to resize them, because the this model expects your images to have equal input shapes.

# ### IMPORTS

# In[ ]:


#import packages
import numpy as np
import re
import os
from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


# ### VARIABLES

# In[ ]:


#paths: paths of all files in the kaggle input directory
paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
#decide on how many dimensions you want to shape your initial data 
latent_dim = 100
#pick a generalized input shape for your data (width, height, channels)
input_shape = (100, 100, 3)
#pick a scaling factor for your upsampling/downsampling operations (convolutional layers)
scale = 2
#pick a kernel size for each model (d_kernel for discriminators, g_kernel for generators)
d_kernel = 3
g_kernel = 4
#decide on how many iterations you want to perform over your data
epochs = 100
#decide on how many units you want your model to process at once
batches = 200
#the directory path of your images
dirpath = "/kaggle/input/natural-images/data/natural_images/fruit/"
#specify how you can query inside the directory
query = "fruit_[0-9]{4}.jpg"
#create a function that returns a normalized 4d-array of images (n_img, width, height, channels) in float32 format
def load_real_samples(paths = paths, dirpath = dirpath, query = query):
	return np.uint8([img_to_array(load_img(x)) for x in re.findall(dirpath + query, ", ".join(paths))]).astype("float32")/255.0


# ### FUNCTIONS

# In[ ]:


# define the standalone discriminator model
def define_discriminator(in_shape=input_shape, kernel=d_kernel, scale=scale):
	model = Sequential()
	model.add(Conv2D(in_shape[0], (kernel,kernel), strides=(scale, scale), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(in_shape[1], (kernel,kernel), strides=(scale, scale), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim=latent_dim, in_shape=input_shape, scale=scale, kernel=g_kernel):
	model = Sequential()
	n_nodes = int(in_shape[0] * in_shape[1] * scale**3)
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((int(in_shape[0]/(scale**2)), int(in_shape[1]/(scale**2)), int(scale**7))))
	model.add(Conv2DTranspose(int(scale**7), (kernel, kernel), strides=(scale,scale), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(int(scale**7), (kernel,kernel), strides=(scale,scale), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(in_shape[2], (int(kernel*2),int(kernel*2)), activation='sigmoid', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	d_model.trainable = False
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# select real samples
def generate_real_samples(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	X = g_model.predict(x_input)
	y = zeros((n_samples, 1))
	return X, y

def save_plot(examples, epoch, n=10):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, :])
	filename = '_generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	X_real, y_real = generate_real_samples(dataset, n_samples)
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	save_plot(x_fake, epoch)
	filename = '_generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=epochs, n_batch=batches):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	for i in range(n_epochs):
		for j in range(bat_per_epo):
			X_real, y_real = generate_real_samples(dataset, half_batch)
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			d_loss, _ = d_model.train_on_batch(X, y)
			X_gan = generate_latent_points(latent_dim, n_batch)
			y_gan = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

            
def gan(latent_dim = latent_dim, input_shape = input_shape, scale = scale, d_kernel = d_kernel, g_kernel = g_kernel, load = load_real_samples, epochs = 100, batches = 200):
    d_model = define_discriminator(kernel = d_kernel, scale = scale, in_shape = input_shape)
    g_model = define_generator(kernel = g_kernel, scale = scale, in_shape = input_shape, latent_dim = latent_dim)
    gan_model = define_gan(g_model, d_model)
    dataset = load()
    train(g_model=g_model, d_model=d_model, gan_model=gan_model, dataset=dataset, latent_dim=latent_dim, n_epochs = epochs, n_batch = batches)


# ### RUN

# In[ ]:


#run the gan model (default=the initialized values above)
gan()

