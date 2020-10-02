#!/usr/bin/env python
# coding: utf-8

# # GAN MODEL 
# 
# Thanks to Dan. The basic structure and functions of this notebook is taken from [Dans' notebok](http://https://www.kaggle.com/dansbecker/running-your-first-gan) and the [TFGAN git repository](http://https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan)
# 
# The notebook is still under progress. I will update(scoring, losses, etc) and explain in more detail in some time. As of now you can still fork and play around by changing things. 

# # Imports

# In[ ]:


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


# ## Generator and Discriminator functions

# In[ ]:


#Define Generator 
def basic_generator(noise):
    """Simple generator to produce MNIST images.

    Args:
        noise: A single Tensor representing noise.

    Returns:
        A generated image in the range [-1, 1].
    """
    channels_after_reshape = 256

    net = Dense(1024, activation='elu')(noise)
    net = Dense(7 * 7 * channels_after_reshape, activation='elu')(net)
    net = Reshape([7, 7, channels_after_reshape])(net)
    net = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation='elu')(net)
    net = Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", activation='elu')(net)
    # Make sure that generator output is in the same range as `inputs`
    # ie [-1, 1].
    net = Conv2D(1, kernel_size=4, activation = 'tanh', padding='same')(net)
    return net



#Define Discriminator 
def basic_discriminator(img, unused_conditioning):
    leaky = LeakyReLU(0.2)
    """Discriminator network on MNIST digits.

    Args:
        img: Real or generated image. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.

    Returns:
        Logits for the probability that the image is real.
    """
    net = Conv2D(64, kernel_size=4, strides=2)(img)
    net = leaky(net)
    net = Conv2D(64, kernel_size=4, strides=2)(net)
    net = leaky(net)
    net = Conv2D(64, kernel_size=4)(net)
    net = leaky(net)

    net = Flatten()(net)
    net = Dense(1024)(net)
    net = leaky(net)
    net = Dense(1, activation='linear')(net)
    return net
 


# ## Function to visualise image 

# In[ ]:


def visualize_training_generator(train_step_num, start_time, plottables, undo_normalization=False):
    """Visualize generator outputs during training.

    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        plottables: Data to plot. Numpy array or list of numpy arrays,
            usually from an evaluated TensorFlow tensor.
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    if type(plottables) == list:
        plottables = np.dstack(plottables)
    plottables = np.squeeze(plottables)
    if undo_normalization:
        plottables = ((plottables * 128) + 128).astype(np.uint8)

    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(plottables)
    plt.show()


def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.
    
    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    queues = tf.contrib.slim.queues
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np))


# # Data Input Pipeline

# In[ ]:


train_fname = '../input/fashionmnist/fashion-mnist_train.csv'
# Size of each image
img_rows, img_cols, depth = 28, 28, 3

#10 labels
num_classes = 10

batch_size = 32

raw = pd.read_csv(train_fname)
num_images = raw.shape[0]
x_as_array = raw.values[:,1:]
# Reshape from 1 vector into an image. Last dimension shows it is greyscale, which is 1 channel
x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
# Optimization with default params is better when vals scaled to [-1, 1]
image_array = ((x_shaped_array - 128)/ 128).astype(np.float32)
# set up target
labels_array = to_categorical(raw.values[:,0], num_classes=10)

# following lines create the iterator/stream of tensors consumed in model training
def dataset_to_stream(inp, batch_size):
    with tf.device('/cpu:0'):
        batched = inp.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        data_feeder = batched.repeat().make_one_shot_iterator().get_next()
    return data_feeder

my_dataset = tf.data.Dataset.from_tensor_slices((image_array))
batched_dataset = dataset_to_stream(my_dataset, batch_size)


# Sanity check that we're getting images.
check_real_digits = tfgan.eval.image_reshaper(
    dataset_to_stream(my_dataset, 20), num_cols=10)
visualize_digits(check_real_digits)


# In[ ]:


batched_dataset.shape


# # Model
# Define the GANModel tuple using the TFGAN library function. For the simplest case, we need the following:
# 
# - A generator function that takes input noise and outputs generated images
# - A discriminator function that takes images and outputs a probability of being real or fake
# - Real images
# - A noise vector to pass to the generator

# In[ ]:


noise_dims = 120
gan_model = tfgan.gan_model(
    basic_generator,
    basic_discriminator,
    real_data=batched_dataset,
    generator_inputs=tf.random_normal([batch_size, noise_dims]))


# # Losses and Optimization
# We next set up the GAN model losses.
# 
# Loss functions are an active area of research. The losses library provides some well-known or successful loss functions, such as the original minimax, Wasserstein, and improved Wasserstein losses.

# In[ ]:


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

# In[ ]:


generator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    improved_wgan_loss,
    generator_optimizer,
    discriminator_optimizer)


# # Set Up Progress Tracking
# 
# This helps us see the evolution in image quality as the GAN is being trained. Specifically, the code below takes a sample of images and shapes them into something that can be viewed.

# In[ ]:


images_to_eval = 10

# For variables to load, use the same variable scope as in the train job.
with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(tf.random_normal([images_to_eval, noise_dims]))

# Reshape eval images for viewing.
generated_data_to_visualize = tfgan.eval.image_reshaper(eval_images[:images_to_eval,...], num_cols=10)


# # Train Steps
#  for-loop for clarity

# In[ ]:



train_step_fn = tfgan.get_sequential_train_steps()

global_step = tf.train.get_or_create_global_step()

n_batches = 30001
with tf.train.SingularMonitoredSession() as sess:
    start_time = time.time()
    for i in range(n_batches):
        train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
        if i % 3000 == 0:
            digits_np = sess.run([generated_data_to_visualize])
            visualize_training_generator(i, start_time, digits_np)

