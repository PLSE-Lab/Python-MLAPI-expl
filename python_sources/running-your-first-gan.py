#!/usr/bin/env python
# coding: utf-8

# # Reminder
# Enable the GPU in Settings.

# # Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Reshape, Conv2DTranspose, LeakyReLU
from keras.utils.np_utils import to_categorical   

# Main TFGAN library.
tfgan = tf.contrib.gan

tf.set_random_seed(0)
tf.reset_default_graph()


# # Utility Functions

# In[2]:


#from learntools.gans.generators import basic_generator
#from learntools.gans.discriminators import basic_discriminator
#from learntools.gans.gan_utils import visualize_training_generator, dataset_to_stream

import sys
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils/gans/')
from generators import basic_generator
from discriminators import basic_discriminator
from gan_utils import visualize_training_generator, dataset_to_stream


# # Data Input Pipeline

# In[3]:


train_fname = '../input/digit-recognizer/train.csv'
# Size of each digit
img_rows, img_cols = 28, 28
# Target has 10 values corresponding to 10 numbers (0, 1, 2 ... 9)
num_classes = 10
# Choice of batch size is not critical
batch_size = 40

raw = pd.read_csv(train_fname)
num_images = raw.shape[0]
x_as_array = raw.values[:,1:]
# Reshape from 1 vector into an image. Last dimension shows it is greyscale, which is 1 channel
x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
# Optimization with default params is better when vals scaled to [-1, 1]
image_array = ((x_shaped_array - 128)/ 128).astype(np.float32)
# set up target
labels_array = to_categorical(raw.values[:,0], num_classes=10)

# following 2 lines create the iterator/stream of tensors consumed in model training
my_dataset = tf.data.Dataset.from_tensor_slices((image_array))
batched_dataset = dataset_to_stream(my_dataset, batch_size)


# # Model
# Define the GANModel tuple using the TFGAN library function. For the simplest case, we need the following:
# 
# - A generator function that takes input noise and outputs generated images
# - A discriminator function that takes images and outputs a probability of being real or fake
# - Real images
# - A noise vector to pass to the generator

# In[4]:


noise_dims = 64
gan_model = tfgan.gan_model(
    basic_generator,
    basic_discriminator,
    real_data=batched_dataset,
    generator_inputs=tf.random_normal([batch_size, noise_dims]))


# # Losses and Optimization
# We next set up the GAN model losses.
# 
# Loss functions are an active area of research. The losses library provides some well-known or successful loss functions, such as the original minimax, Wasserstein, and improved Wasserstein losses.

# In[5]:


# Example of classical loss function.
#vanilla_gan_loss = tfgan.gan_loss(
#    gan_model,
#    generator_loss_fn=tfgan.losses.minimax_generator_loss,
#    discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)

# Wasserstein loss (https://arxiv.org/abs/1701.07875) with the 
# gradient penalty from the improved Wasserstein loss paper 
# (https://arxiv.org/abs/1704.00028).
improved_wgan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty_weight=1.0)


# ### Optimizer Settings
# 
# The choice of optimizer and settings has been the subject of a lot of guesswork and iteration. When getting started, it's likely not a good use of time to fiddle with these. This also may be 

# In[6]:


generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    improved_wgan_loss,
    generator_optimizer,
    discriminator_optimizer)


# # Set Up Progress Tracking
# 
# This helps us see the evolution in image quality as the GAN is being trained. Specifically, the code below takes a sample of images and shapes them into something that can be viewed.

# In[7]:


images_to_eval = 20

# For variables to load, use the same variable scope as in the train job.
with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(tf.random_normal([images_to_eval, noise_dims]))

# Reshape eval images for viewing.
generated_data_to_visualize = tfgan.eval.image_reshaper(eval_images[:images_to_eval,...], num_cols=10)


# # Train Steps
# 
# Now we're ready to train. TFGAN handles the alternating training scheme. There is a gan_train function which provies more slightly more automation, but we will run the training in a for-loop so we can introspect more.

# In[ ]:


g_d_updates_per_step = tfgan.GANTrainSteps(1,2)  # do 1 gen step, then 2 disc steps.  
train_step_fn = tfgan.get_sequential_train_steps(g_d_updates_per_step)

global_step = tf.train.get_or_create_global_step()

n_batches = 1501
with tf.train.SingularMonitoredSession() as sess:
    start_time = time.time()
    for i in range(n_batches):
        train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
        if i % 100 == 0:
            digits_np = sess.run([generated_data_to_visualize])
            visualize_training_generator(i, start_time, digits_np)

