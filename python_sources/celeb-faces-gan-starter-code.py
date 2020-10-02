#!/usr/bin/env python
# coding: utf-8

# # Celebrity Faces DCGAN Starter Code
# This is starter code for a Deep Convolutional Generative Adversarial Network (DCGAN) to generate celeb faces.  I did heavy code reuse/modification of the following tensorflow turtorial:
# https://www.tensorflow.org/beta/tutorials/generative/dcgan
# 
# This tensorflow tutorial also gave me some help:
# https://www.tensorflow.org/tutorials/load_data/images#build_a_tfdatadataset
# 
# I plan to update this Notebook to improve (1) documentation, (2) performance, and (3) visualization.
# 
# Feel free to leave comments for improvements!

# In[ ]:


import warnings # We'll use this to suppress warnings caused by TensorFlow
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # generating plot

import tensorflow as tf # modeling/training
tf.enable_eager_execution() # Must execute this at the beginning of the code
                            # See https://www.tensorflow.org/guide/eager for details
import time # Used for epoch timing

import imageio # GIF generation
import glob # GIF generation
import PIL # GIF generation


import os
data_dir = '/kaggle/input/celeba-dataset/'
os.listdir(data_dir)


# We will not actually use any of the data in the CSV files, only the names of the images (for now).

# In[ ]:


list_eval_partition = pd.read_csv(data_dir + 'list_eval_partition.csv')
names_df = list_eval_partition['image_id']
names_df.head()


# Let's take a random sample of our images to see what they look like.  I also took a sample of the image shapes to make sure they are all the same. (I believe they are all [218, 178, 3]).

# In[ ]:


img_names = names_df.sample(n=16).values
shapes = []
plt.figure(figsize=(10,10))
for i, name in enumerate(img_names):
    plt.subplot(4, 4, i + 1)
    img = plt.imread(data_dir + 'img_align_celeba/img_align_celeba/' + name)
    shapes.append(img.shape)
    plt.imshow(img)
    plt.title(name)
    plt.axis('off')
_=plt.suptitle('')


# We will use this function to load the image, resize it to the size necessary for our models, and then preprocess it (currently only centering and rescaling).  Im resizing based on what I need for the models - it'll become more clear why we are resizing when you see the models.

# In[ ]:


def load_and_preprocess_image(name):
    image = tf.io.read_file(data_dir + 'img_align_celeba/img_align_celeba/' + name)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_images(image, (216, 176))
    image = (image - 127.5) / 127.5
    return image


# Below is an important step to ensure all of our images are represented by a Dataset.  I referenced this Tensorflow tutorial as mentioned above: https://www.tensorflow.org/tutorials/load_data/images#build_a_tfdatadataset

# In[ ]:


BATCH_SIZE = 256
name_ds = tf.data.Dataset.from_tensor_slices(names_df.values)
image_ds = name_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
image_ds = image_ds.shuffle(buffer_size=2000).batch(BATCH_SIZE)

num_batches = int(np.ceil(len(names_df) / BATCH_SIZE))
print('There are {} batches'.format(num_batches))


# ### Building the models
# Now we need to build our two models - the generator and the discriminator.  First, the generator will take a random input of length 100 and generate it into a face (after training of course).
# 
# I used this tutorial as a baseline: https://www.tensorflow.org/beta/tutorials/generative/dcgan
# 
# You can skip the section below if you feel comfortable with how the model upsamples and why we chose to resize our images.

# ### Generator model dimensions
# The tricky part with the generator model is the dimensions of each step of the model. Remember, we are taking an input of shape (BATCH_SIZE, 100) and upsampling it to an output shape of (BATCH_SIZE, x_output, y_output, 3).  In this case, I made the x_output and y_output dimensions as close to the original image shapes as possible.
# 
# In order to upsample we need to essnetialy scale up our pixels by an integer at each step.  To do this we need to find the prime factors of our x_output and y_output dimensions.  Notice that 218 and 178 only have two prime factors each (2 x 109 and 2 x 89, respecively).  I chose output dimensions of 216 and 176, which factor nicely to primes of 2 x 2 x 2 x 3 x 3 x 3 and 2 x 2 x 2 x 2 x 11, respectively.
# 
# Now we can use these prime factors to decide the dimensions of the model at each stage.  The stride lengths of each Conv2DTranspose step multiplies each x_output and y_output dimension.  For example: working backwards in the x_output dimension, we take 216 / 2 = 108 for our last stage, 108 / 3 = 36 for the middle stage, and 36 / 3 = 12 for the first stage.

# In[ ]:


from tensorflow.keras import layers
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(12*11*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((12, 11, 128)))
    assert model.output_shape == (None, 12, 11, 128)

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(3, 4), padding='same', use_bias=False))
    assert model.output_shape == (None, 36, 44, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(3, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 108, 88, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 216, 176, 3)

    return model


# Let's check to see a summary of our generator model and a randomly generated image.  A couple of things here: First, obviously we haven't yet trained our model, so we are not going to get anything that resembles a face. Second, we need to make sure we reverse the preprocessing that we would have done previously to the input.

# In[ ]:


generator = make_generator_model()
noise_image = tf.random.normal([1,100,])
generated_image = generator(noise_image, training=False)
plt.imshow((generated_image[0]*127.5 +127.5) / 255.)
_=plt.axis('off')
generator.summary()


# The discriminator model is much more straight forward.  We are simply going to be doing binary classification (real or fake).  In this case we are just downsampling from (BATCH_SIZE, x_output, x_output, 3) to a single number between 0 (fake) and 1 (real).

# In[ ]:


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[216, 176, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# Let's make sure our discriminator model gives us that single number given our randomly generated image above.

# In[ ]:


discriminator = make_discriminator_model()
discriminator.summary()
print(discriminator(generated_image))


# ### Loss functions
# Now it's time to take a step back and think about what the two models are really trying to accomplish.  Remember, as stated above, the output of our discriminator is a number between zero (fake image) and one (real image).
# 
# The discriminator model is trying to classify all of the fake images that the generator makes as fake and all of the true images as real.  Therefore, it's loss function will look at the cross-entropy between group of ones and the real output AND a group of zeros and the fake output.
# 
# The generator model is trying to make a fake image that looks like a real one.  Therefore, it's loss function will look at the cross-entropy between a group of ones and the fake output.

# In[ ]:


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_obj(tf.ones_like(real_output), real_output)
    fake_loss = loss_obj(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return loss_obj(tf.ones_like(fake_output), fake_output)


# We'll simply use a Adam optimizer for each model.

# In[ ]:


gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)


# ## Training
# Here's where the rubber meets the road.  We need to define what happens with each batch of images on a training step.  The train_step() function performs the following steps:
# 1. Make a vector of shape (BATCH_SIZE, 100) of random numbers
# 2. Generate a fake image based on the random vector
# 3. Predict whether or not the real images are real and the fake images are fake
# 4. Compute the loss for each model (see above)
# 5. Compute the gradients of the loss with respect to the trainable variables of each model
# 6. Update our trainable variables based on the gradient
# 
# We will return the calculated loss for each model in order to visualize the training process

# In[ ]:


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
        
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    return gen_loss, disc_loss


# This function will help us visualize and save a group of generated images to visualize how our model is doing.  Again, remember we'll have to 'undo' the preprocessing that we did at the beginning.

# In[ ]:


def show_and_generate_images(model, epoch, test_output):
    predictions = model(test_output, training=False)
    
    plt.figure(figsize=(10,10))
    for i in range(len(test_output)):
        plt.subplot(4,4,i+1)
        plt.imshow((predictions[i] * 127.5 +127.5) / 255.)
        plt.axis('off')
        
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    _=plt.show()


# Putting it all together, for each epoch, we will itterate over the dataset performing a training step for each batch.  The history DataFrame is used for post training analysis of the average loss of each model at each epoch.

# In[ ]:


def train(dataset, epochs):
    print('Begining to train...')
    
    history = pd.DataFrame(['gen_loss', 'disc_loss'])
    for epoch in range(epochs):
        start = time.time()
        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()
        for i, images in enumerate(dataset):
            gen_loss, disc_loss = train_step(images)
            epoch_gen_loss.update_state(gen_loss)
            epoch_disc_loss.update_state(disc_loss)

        show_and_generate_images(generator, epoch + 1, seed)
        stats = 'Epoch {0} took {1} seconds. Gen_loss: {2:0.3f}, Disc_loss: {3:0.3f}'
        print(stats.format(epoch + 1, int(time.time() - start), 
                           epoch_gen_loss.result().numpy(), 
                           epoch_disc_loss.result().numpy()))
        history = history.append({'gen_loss': epoch_gen_loss.result().numpy(), 
                                  'disc_loss': epoch_disc_loss.result().numpy()}, 
                                  ignore_index=True)
        
    return history


# Train the models!
# We use a 'seed' tensor to show the same images over training time (we'll use it in a nice GIF after training).
# For our history DataFrame, we increment the index to start at epoch 1.
# 
# **WARNING: this code takes ~6 hours to run.**

# In[ ]:


EPOCHS = 32
seed = tf.random.normal([16, 100])
history = train(image_ds, EPOCHS)
history.index = history.index + 1


# ### Post-training analysis
# To visualize the training process, we plot the loss of each model.  This should be a prety intersting graphic, because it shows how well the discriminator model is discriminating versus how well the generator model is generating.

# In[ ]:


ax = plt.axes(xlabel='epoch', ylabel='loss')
history.plot(ax=ax, figsize=(10,7))
_=plt.title('Loss History')


# Finally, we re-use some code from the Tensorflow DCGAN to show the GIF of our random seed over training steps.

# In[ ]:


anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
  IPython.display.Image(filename=anim_file)

