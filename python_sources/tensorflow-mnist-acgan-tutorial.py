#!/usr/bin/env python
# coding: utf-8

# Adapted from [GitHub ACGAN Keras Example](https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py) and [GitHub Training Override Example](https://github.com/keras-team/keras-io/blob/master/examples/generative/dcgan_overriding_train_step.py)
# 
# TensorFlow 1.x --> Tensorflow 2.x
# 
# Uses GPU Accelerator

# ## 1. Introduction
# 
# This tutorial will go over the steps to build an ACGAN model on the MNIST dataset. The MNIST dataset is a dataset of images of handwritten digits 0 through 9. The ACGAN model, once trained, will generate fake images that resemble the real images of each class.
# 
# Run the following cells to import the necessary packages for this tutorial.

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from collections import defaultdict
from PIL import Image
from six.moves import range

print("Tensorflow version " + tf.__version__)


# For this tutorial, we will be focusing on the MNIST dataset. The MNIST dataset is a dataset of images of handwritten digits from 0 to 9. Since there are 10 digits, there are also 10 classes. The random seed is used so that the results are reproducible. We will load in our data after we define our methods below.

# In[ ]:


np.random.seed(1337)
num_classes = 10

epochs = 30
latent_dim = 128

adam_lr = 0.0002
adam_beta_1 = 0.5


# ## 2. Building the generator
# 
# The first step of building the GAN is to build the generator model. As the name implies, the generator model will be the part of our model that generates the images. The function that builds our generator is defined in the following cell using the TensorFlow Keras API.

# In[ ]:


def build_generator(latent_size):
    cnn = tf.keras.Sequential()

    cnn.add(layers.Dense(7 * 7 * 128, input_dim=latent_size))
    cnn.add(layers.LeakyReLU(alpha=0.2))
    cnn.add(layers.Reshape((7, 7, 128)))

    cnn.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same',
                          kernel_initializer='glorot_normal'))
    cnn.add(layers.LeakyReLU(alpha=0.2))
    cnn.add(layers.BatchNormalization())

    cnn.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same',
                          kernel_initializer='glorot_normal'))
    cnn.add(layers.LeakyReLU(alpha=0.2))
    cnn.add(layers.BatchNormalization())

    cnn.add(layers.Conv2D(1, 7, padding='same',
                          activation='tanh',
                          kernel_initializer='glorot_normal'))
    
    return cnn


# ## 3. Build the discriminator
# 
# The second part of our generative model is our discriminator. The discriminator will be trained separately. It determines if the inputted image is a real or a fake image. The discriminator is used to train the generator.

# In[ ]:


def build_discriminator():

    cnn = tf.keras.Sequential()

    cnn.add(layers.Conv2D(32, 3, padding='same', strides=2,
                          input_shape=(28, 28, 1)))
    cnn.add(layers.LeakyReLU(0.2))
    cnn.add(layers.Dropout(0.3))

    cnn.add(layers.Conv2D(64, 3, padding='same', strides=1))
    cnn.add(layers.LeakyReLU(0.2))
    cnn.add(layers.Dropout(0.3))

    cnn.add(layers.Conv2D(128, 3, padding='same', strides=2))
    cnn.add(layers.LeakyReLU(0.2))
    cnn.add(layers.Dropout(0.3))

    cnn.add(layers.Conv2D(256, 3, padding='same', strides=1))
    cnn.add(layers.LeakyReLU(0.2))
    cnn.add(layers.Dropout(0.3))
    
    cnn.add(layers.GlobalMaxPooling2D()),
    cnn.add(layers.Dense(1))
    
    return cnn


# Run the following cells to train the discriminator.

# In[ ]:


print('Discriminator model:')
discriminator = build_discriminator()
discriminator.summary()


# ## 3. Build the combined model
# 
# Although the discriminator was trained separately, it's necessary to train the generator by using the discriminator. The generator will output an image and the discriminator will determine if the generated fake image is real or fake. The output of the discriminator will help train the generator.

# In[ ]:


print('Generator model:')
generator = build_generator(latent_dim)
generator.summary()


# We are going to create a new class called GAN and it will be a subclass of the TensorFlow Keras Model class. By creating a new subclass, we can rewrite the ```train_step``` function, which will allow us to call ```model.fit()```. This reduces the code that we have to write to train this generative model and it allows for ease of readability.

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


# We will also create a subclass of the TensorFlow Keras Callback class called GANMonitor. Calling an instance of this subclass allos us to save a generated image at the end of each epoch to see how the model is improving.

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


# Let's now build our combined model. The generator is the model will be trained with the output of the discriminator. Additionally, since we have two ouputs, we want to measure both the binary crosentropy and the sparse cateogircal crossentropy as our losses.

# In[ ]:


gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
)


# ## 4. Load the data
# 
# Now that the models have been defined and built, we have to load the data to train the model on. This tutorial will focus on the MNIST dataset. Luckily for us, the MNIST dataset can be easily accessed from the TensorFlow API.
# 
# We want our data to be normalized to [0, 1]. As the data initially is between [0, 255], we will have to do some basic preprocessing and reshaping. Additionally, don't need a training and testing dataset because the images in both the the training and testing dataset are real images. Therefore, we can combine the two into a singular dataset to be used in our training.
# 
# Run the next cell to load and normalize the data.

# In[ ]:


batch_size = 64
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)


# ## 5. Train the model
# 
# We previously specified the model to train for 30 epochs. Try training the model on more or less epochs to see the differences in loss and accuracy. The loss for our generative model and for our discriminator can be seen at the end of each epoch.

# In[ ]:


gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim)]
)


# ## 6. Visualize the generated images
# 
# Run the following cell to visualize some of the saved generated images.

# In[ ]:


get_ipython().system('ls')


# In[ ]:


Image.open("generated_img_2_20.png")


# In[ ]:




