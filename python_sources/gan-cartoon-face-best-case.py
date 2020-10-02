#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q tf-nightly')


# In[ ]:


import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import pathlib
import random
import IPython.display as display

from IPython import display


# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[ ]:


cartoon_image_path = os.listdir('../input/cartoonfacedatasetzip/cartoon_face_dataset/cartoonset10k')
cartoon_image_path = [os.path.join("../input/cartoonfacedatasetzip/cartoon_face_dataset/cartoonset10k", x) for x in cartoon_image_path]

cim = []
for x in cartoon_image_path:
    if x[-3:]=='png':
        cim.append(x)
cartoon_image_path = cim
len(cartoon_image_path)


# In[ ]:


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    #image = ndi.gaussian_filter(image, 5)
    image = tf.image.resize_images(image, [58, 58])
    image = tf.image.central_crop(central_fraction=0.55, image=image)
    #image = tf.image.rgb_to_grayscale(image)
    image -= 127.5
    image /= 127.5  # normalize to [-1,1] range
    return image

def load_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


# In[ ]:


cartoon_path_ds = tf.data.Dataset.from_tensor_slices(cartoon_image_path)
cartoon_image_ds = cartoon_path_ds.map(load_image, num_parallel_calls=AUTOTUNE)


# In[ ]:


def show_images(images, l=2, epoch=-1):
    plt.figure(figsize=(l*1.5,l*1.5))
    for n,image in enumerate(images):
        image = (image + 1 ) / 2 #scale to [0,1]
        plt.subplot(l,l,n+1)
        plt.imshow(image)
        plt.grid(False)
    plt.show()
    print(image.shape)
    if epoch!=-1:
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
show_images(cartoon_image_ds.take(4))


# In[ ]:


cartoon_BATCH_SIZE = 32
cartoon_ds = cartoon_image_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(cartoon_image_path)))
cartoon_ds = cartoon_ds.batch(cartoon_BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
cartoon_ds = cartoon_ds.prefetch(buffer_size=AUTOTUNE)
cartoon_ds


# In[ ]:


for cartoon in cartoon_ds:
    test_cartoon = cartoon
    break
    
print(test_cartoon.shape)


# In[ ]:


test_cartoon = test_cartoon[:4]


# In[ ]:


show_images(test_cartoon)


# In[ ]:


print(np.max(test_cartoon))
print(np.min(test_cartoon))


# In[ ]:


def make_generator_model():
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4*4*512, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Reshape((4, 4, 512)))
    
    model.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return model


# In[ ]:


generator = make_generator_model()
generator.summary()


# In[ ]:


noise_dim = 100
noise = tf.random_normal([1, noise_dim])
gen_out = generator(noise)
show_images(gen_out, l=1)


# In[ ]:


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(32,32,3)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
      
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
       
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    #model.add(tf.keras.layers.Activation('sigmoid'))

    return model


# In[ ]:


discriminator = make_discriminator_model()
discriminator.summary()


# In[ ]:


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


# In[ ]:


def discriminator_loss(real_out, generated_out):
    real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_out), real_out)
    generated_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(generated_out), generated_out)
    total_loss = real_loss + generated_loss
    return total_loss


# In[ ]:


g_optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
d_optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)


# In[ ]:


2.5e-4


# In[ ]:


noise_dim = 100
num_examples_to_generate = 16

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 noise_dim])


# In[ ]:


'''def train_step(cartoon_images):
    
    noises = tf.random_normal([cartoon_BATCH_SIZE, noise_dim])
    for noise, cartoon_image in zip(noises, cartoon_images):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:

            # generating noise from a normal distribution

            generated_image = generator(noise, training = True)

            real_out = discriminator(cartoon_image, training =True)
            generated_out = discriminator(generated_image, training = True)

            gen_loss = generator_loss(generated_out)
            dis_loss = discriminator_loss(real_out, generated_out)

        gen_grad = gen_tape.gradient(gen_loss, generator.variables)
        dis_grad = dis_tape.gradient(dis_loss, discriminator.variables)

        g_optimizer.apply_gradients(zip(gen_grad, generator.variables))
        d_optimizer.apply_gradients(zip(dis_grad, discriminator.variables))
'''


# In[ ]:


def train_step(cartoon_images):
            
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:

        noise = tf.random_normal([cartoon_BATCH_SIZE, noise_dim])
        # generating noise from a normal distribution
        generated_images = generator(noise, training = True)

        real_out = discriminator(cartoon_images, training = True)
        generated_out = discriminator(generated_images, training = True)

        gen_loss = generator_loss(generated_out)
        dis_loss = discriminator_loss(real_out, generated_out)

    gen_grad = gen_tape.gradient(gen_loss, generator.variables)
    dis_grad = dis_tape.gradient(dis_loss, discriminator.variables)

    g_optimizer.apply_gradients(zip(gen_grad, generator.variables))
    d_optimizer.apply_gradients(zip(dis_grad, discriminator.variables))


# In[ ]:


train_step = tf.contrib.eager.defun(train_step)


# In[ ]:


def generate_test_prediction(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    
    gen_out = generator(test_input, training=False)
    show_images(gen_out, 4, epoch=epoch)
    #print(discriminator(gen_out[:4]))
    #print(discriminator(test_cartoon))
    #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


# In[ ]:


gen_out = generator(random_vector_for_generation, training=False)
print(np.max(gen_out))
print(np.min(gen_out))


# In[ ]:


generate_test_prediction(generator, 4, random_vector_for_generation)


# In[ ]:


def train(epochs):
    epoch = 0
    start = time.time()
    for cartoon_images in cartoon_ds:
        train_step(cartoon_images)
        epoch = epoch + 1
        if(epoch%10==0):
            #display.clear_output(wait = True)
            generate_test_prediction(generator, epoch, random_vector_for_generation)
        print ('Time taken for epoch {} is {} sec'.format(epoch, time.time()-start))
        start = time.time()
        if epoch == epochs:
            break


# In[ ]:


EPOCHS = 100


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train(EPOCHS)')


# In[ ]:


gen_out = generator(random_vector_for_generation)
print(np.max(gen_out))
print(np.min(gen_out))


# In[ ]:


for n,image in enumerate(gen_out):
    image = image/2 + 0.5
    plt.subplot(4,4,n+1)
    plt.imshow(image)
    plt.grid(False)


# In[ ]:




