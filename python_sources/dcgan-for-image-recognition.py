# %% [code]
from __future__ import absolute_import, division, print_function, unicode_literals

# %% [code]
try:
    %tensorflow_version 2.x
except Exception:
    pass

# %% [code]
import tensorflow as tf
tf.__version__

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
!pip install imageio

# %% [code]
import glob
import imageio
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras import layers, activations
import time
import cv2
import wandb
import numpy as np
from IPython import display

! wandb login ab8dc273117f79beebc0097569afc05902f20b1e

# %% [code]
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# %% [code]
images = os.listdir("../input/celeba-dataset/img_align_celeba/img_align_celeba/")
trainData = []
# print(type(images))
for i in range(30):
    trainData.append(cv2.resize(cv2.imread("../input/celeba-dataset/img_align_celeba/img_align_celeba/"+images[i]),(64,64)))
#(training examples. height, width, channels)
trainData = np.float32((np.array(trainData) - 127.5)/127.5)
#(218,178,3)
# print(trainData.shape)
trainDataset = tf.data.Dataset.from_tensor_slices(trainData).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# %% [code]
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((4,4,1024)))
    
    model.add(layers.Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    
    return model

# %% [code]
generator = make_generator_model()

noise = tf.random.normal([10, 100])
generated_image = generator(noise,  training=False)

plt.imshow(generated_image[0, :, :, 0])

# %% [code]
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128,(4,4), strides=(2,2), padding='same', input_shape=[64,64,3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256,(4,4), strides=(2,2), padding = 'same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(512,(4,4), strides=(2,2), padding = 'same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(1024,(4,4), strides=(2,2), padding = 'same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(1,(4,4), strides=(1,1), padding = 'same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

# %% [code]
# Define the loss and optimizers
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# %% [code]
# Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# %% [code]
# Generator loss
def generator_loss(fake_output):
    gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return gen_loss

# %% [code]
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# %% [code]
discriminator = make_discriminator_model()
decision = discriminator(generated_image)

# %% [code]
EPOCHS = 300
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# %% [code]
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss
    

# %% [code]
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("epoch {}".format(epoch), "time:", start)
        wandb.log({'epoch': epoch, 'time': start})
        # something wrong here
        for i in dataset:
            gen_loss, disc_loss = train_step(dataset)
            print(gen_loss, disc_loss)
            noise = tf.random.normal([10,100])
            generator(noise, training=False)
            generated = generator(noise, training=False)
            tensor = generated[0,:,:,0]
            tensorInArr = tf.keras.backend.get_value(tensor)
            plt.imshow(tensorInArr)
            wandb.log({'gen_loss': gen_loss, 'disc_loss': disc_loss, 'photo': plt})
        plt.show()
        

# %% [code]
    wandb.init(project="visualize-models")
    wandb.config.batch_size = BATCH_SIZE


# %% [code]
# def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
#  predictions = model(test_input, training=False)

#  fig = plt.figure(figsize=(4,4))

#  for i in range(predictions.shape[0]):
 #     plt.subplot(4, 4, i+1)
  #    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
   #   plt.axis('off')

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

# %% [code]
train(trainData, EPOCHS)