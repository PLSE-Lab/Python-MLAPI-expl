#!/usr/bin/env python
# coding: utf-8

# Adapted from [Machine Learning Mastery Article](https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/) and [Keras Training Override Example](https://github.com/keras-team/keras-io/blob/master/examples/generative/dcgan_overriding_train_step.py)
# 
# 
# TensorFlow 1.x --> TensorFlow 2.x

# # Introduction
# 
# This tutorial will go over how to train an ACGAN (auxiliary classifier generative adversarial network) using the Fashion MNIST dataset. An ACGAN model will generate images for a given condition, with the condition being a class.
# 
# The Fashion MNIST dataset is a datset consistng of 60000 training examples. Each example is a 28x28 image of an article of clothing, with each example falling under one of 10 classes.
# 
# The classes are as follows:
# 0. T-shirt
# 1. Trouser
# 2. Pullover
# 3. Dress
# 4. Coat
# 5. Sandal
# 6. Shirt
# 7. Sneaker
# 8. Bag
# 9. Ankle boot
# 
# Run the following cell to download the necessary packages. If running on a TPU, change the ```TPU_used``` variable to true. Else, change the accelerator on the right to GPU.
# 

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time
from PIL import Image

from keras.datasets.fashion_mnist import load_data

print(tf.__version__)

TPU_used = False

if TPU_used:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# The random seed allows for the notebook to be reproducible. As there are the 10 classes in the Fashion MNIST dataset, we set the number of classes to to 10. The number of epochs is initially set to 30 but feel free to change the number of epochs. The learning rate and beta value will be used later when we compile our models.

# In[ ]:


np.random.seed(1337)
num_classes = 10

epochs = 30
latent_dim = 128

adam_lr = 0.0002
adam_beta_1 = 0.5


# ## 1. Importing data
# 
# The following cell defines a function to import the Fashion MNIST dataset. As we are building a generative model, we don't need a separate testing set and all the images in the datset will be used as real images for our model.
# 
# The data is scaled so that the data is normalized to [0, 1] rather than [0, 255]. Normalizing the data is an important part of preprocessing.

# In[ ]:


batch_size = 64
(x_train, _), (x_test, _) = load_data()
all_images = np.concatenate([x_train, x_test])
all_images = all_images.astype("float32") / 255
all_images = np.reshape(all_images, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_images)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)


# ## 2. Building the discriminator
# 
# To build a GAN, a discriminator model must first be built. The discriminator takes an image in as its input and returns the probability that the image is real.
# 
# We define the method to build our discriminator in the following cell using the TensorFlow Keras API.
# 
# Conv2D is a convolution layer that applies a filter of a specified size (in this example, a 3x3 kernel, on each pixel). This allows for certain features to stand out.
# 
# LeakyRelu is similar to a rectifier, but instead of having all negative values become 0, there is a small negative slope. This layer allows the model to find nonlinearities.
# 
# Dropout randomly ignores 0.5 of the input nodes. This prevents overfitting  by forcing to model to learn new features.

# In[ ]:


def define_discriminator():
    model = tf.keras.Sequential(
        [
            layers.Conv2D(32, 3, strides=2, padding='same',
                          input_shape=(28, 28, 1)),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.5),
            
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.5),
            
            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.5),
            
            layers.Conv2D(256, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.5),
            
            layers.GlobalMaxPooling2D(),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    
    return model


# In[ ]:


if TPU_used:
    with tpu_strategy.scope():
        discriminator = define_discriminator()
else:
    discriminator = define_discriminator()
discriminator.summary()


# ## 4. Building the generator
# 
# Like the discriminator, the generator must also be built before we can move on to our GAN model.
# 
# The generator take in a random point from the latent space and a class label, and it returns a generated image that falls under the specified class label. A latent space is a way to represent condensed information in a way that similar datapoints have smaller distances between them.
# 
# A point in the latent space can by used to create multiple 7x7 feature maps, and these maps, along with the feature map created by the class label, can be upcaled to a 14x14 and then a 28x28 image.
# 
# Because the generator is trained with the discriminator, it should not be compiled.

# In[ ]:


def define_generator(latent_size):
    model = tf.keras.Sequential(
        [
            layers.Dense(7 * 7 * 128, input_dim=latent_size),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, 128)),
            
            layers.Conv2DTranspose(128, 4, strides=2, padding='same',
                                   kernel_initializer='glorot_normal'),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Conv2DTranspose(128, 4, strides=2, padding='same',
                                   kernel_initializer='glorot_normal'),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            
            layers.Conv2D(1, 7, padding='same',
                          activation='tanh',
                          kernel_initializer='glorot_normal')
        ]
    )
    
    return model


# In[ ]:


if TPU_used:
    with tpu_strategy.scope():
        generator = define_generator(latent_dim)
else:
    generator = define_generator(latent_dim)
generator.summary()


# ## 5. Buiding the composite model.
# 
# As mentioned previously, the generator is trained using the discriminator model. The composite model will take in the same input as the generator, feed it into the generator, and the generated image will be fed into the discriminator. During training, we do not want to update the weights in the discriminator. The discriminator will be trained separately from within the composite model.
# 
# We are going to create a new class called GAN and it will be a subclass of the TensorFlow Keras Model class. By creating a new subclass, we can rewrite the train_step function, which will allow us to call ```model.fit()```. This reduces the code that we have to write to train this generative model and it allows for ease of readability.

# In[ ]:


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


# We will also create a subclass of the TensorFlow Keras Callback class called GANMonitor. Calling an instance of this subclass allows us to save a generated image at the end of each epoch to see how the model is improving.

# In[ ]:


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))


# Let's now build our combined model.

# In[ ]:


if TPU_used:
    with tpu_strategy.scope():
        gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE),
        )
else:
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
        loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )


# ## 6. Train the model
# 
# The following function defines how the model is going to be trained. It is trained with a batch, half of which is real images and the other half is fake images created by the generator.
# 
# Run the following cell to train the model.

# In[ ]:


gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim)]
)


# ## 7. Visualize images using created by the model

# In[ ]:


get_ipython().system('ls')


# In[ ]:


Image.open("generated_img_2_20.png")


# In[ ]:




