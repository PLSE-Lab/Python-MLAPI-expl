#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shutil
import os

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D,     BatchNormalization, UpSampling2D, Reshape, Conv2DTranspose, ReLU
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et


INPUT_SIZE = 100
PLOT_FRECUENCY = 50


# # Image Loading and Processing

# In[ ]:


def read_image(file, bounds):
    image = open_image(file, bounds)
    image = normalize_image(image)
    return image


def open_image(file, bounds):
    image = Image.open(file)
    image = image.crop(bounds)
    image = image.resize((64, 64))
    return np.array(image)


# Normalization, [-1,1] Range
def normalize_image(image):
    image = np.asarray(image, np.float32)
    image = image / 127.5 - 1
    return img_to_array(image)


# Restore, [0,255] Range
def denormalize_image(image):
    return ((image+1)*127.5).astype(np.uint8)


def load_images():
    images = []

    for breed in os.listdir('../input/annotation/Annotation/'):
        for dog in os.listdir('../input/annotation/Annotation/' + breed):
            tree = et.parse('../input/annotation/Annotation/' + breed + '/' + dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                box = o.find('bndbox')
                xmin = int(box.find('xmin').text)
                ymin = int(box.find('ymin').text)
                xmax = int(box.find('xmax').text)
                ymax = int(box.find('ymax').text)

            bounds = (xmin, ymin, xmax, ymax)
            try:
                image = read_image('../input/all-dogs/all-dogs/' + dog + '.jpg', bounds)
                images.append(image)
            except:
                print('No image', dog)

    return np.array(images)


x_train = load_images()


# # Adversarial Networks

# ## Generator

# In[ ]:


def create_generator():
    generator = Sequential()
    generator.add(Dense(units=256*4*4,input_dim=INPUT_SIZE))
    generator.add(Reshape((4,4,256)))

    generator.add(Conv2DTranspose(1024, 4, strides=1, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(512, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(256, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())

    generator.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(64, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(3, 3, strides=1, activation='tanh', padding='same'))
    
    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))

    return generator


generator = create_generator()
generator.summary()


# ## Discriminator

# In[ ]:


def create_discriminator():
    discriminator = Sequential()

    discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding='same', input_shape=(64,64,3)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(1, kernel_size=4, strides=1, padding='same'))

    discriminator.add(Flatten())
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))
    return discriminator


discriminator = create_discriminator()
discriminator.summary()


# ## GAN

# In[ ]:


def create_gan(generator, discriminator):
    discriminator.trainable = False

    gan_input = Input(shape=(INPUT_SIZE,))
    generator_output = generator(gan_input)
    gan_output = discriminator(generator_output)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))

    return gan


gan = create_gan(generator, discriminator)
gan.summary()


# # Plotting

# In[ ]:


def plot_images(generator, size=25, dim=(5,5), figsize=(10,10)):
    noise= generate_noise(size)
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(denormalize_image(generated_images[i]), interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_loss(epoch, g_losses, d_losses):
    plt.figure(figsize=(10,5))
    plt.title("Loss, Epochs 0-" + str(epoch))
    plt.plot(g_losses,label="Generator")
    plt.plot(d_losses,label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# # Training

# In[ ]:


def generate_noise(size):
    return np.random.normal(0, 1, size=[size, INPUT_SIZE])


def training(epochs=1, batch_size=32):
    #Loading Data
    batches = x_train.shape[0] / batch_size
    
    # Adversarial Labels
    y_valid = np.ones(batch_size)*0.9
    y_fake = np.zeros(batch_size)
    discriminator_loss, generator_loss = [], []

    for epoch in range(1, epochs+1):
        g_loss = 0; d_loss = 0

        for _ in range(int(batches)):
            # Random Noise and Images Set
            noise = generate_noise(batch_size)
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate Fake Images
            generated_images = generator.predict(noise)
            
            # Train Discriminator (Fake and Real)
            discriminator.trainable = True
            d_valid_loss = discriminator.train_on_batch(image_batch, y_valid)
            d_fake_loss = discriminator.train_on_batch(generated_images, y_fake)            

            d_loss += (d_fake_loss + d_valid_loss)/2
            
            # Train Generator
            noise = generate_noise(batch_size)
            discriminator.trainable = False
            g_loss += gan.train_on_batch(noise, y_valid)
            
        discriminator_loss.append(d_loss/batches)
        generator_loss.append(g_loss/batches)
            
        if epoch % PLOT_FRECUENCY == 0:
            print('Epoch', epoch)
            plot_images(generator)
            plot_loss(epoch, generator_loss, discriminator_loss)

    
training(epochs=200)


# # Submission

# In[ ]:


def save_images(generator):
    if not os.path.exists('../output'):
        os.mkdir('../output')

    noise = generate_noise(10000)
    generated_images = generator.predict(noise)

    for i in range(generated_images.shape[0]):
        image = denormalize_image(generated_images[i])
        image = array_to_img(image)
        image.save( '../output/' + str(i) + '.png')

    shutil.make_archive('images', 'zip', '../output')
    
    
save_images(generator)

