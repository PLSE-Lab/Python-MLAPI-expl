#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil

import warnings
warnings.filterwarnings("ignore")

from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn

import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
from keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

INPUT_SIZE = 100
PLOT_FRECUENCY = 10


# In[ ]:


#array de carpetas con las imagenes de cada raza
razas = os.listdir("../input/stanford-dogs-dataset/images/Images/") 

#cantidad de carpetas de razas
cantidadDeRazas = len(razas) 
print("Hay {} Razas".format(cantidadDeRazas))

#contador del total de todas las imagenes en la data
numeroTotalDeImagenes = 0 
for raza in razas:
    numeroTotalDeImagenes += len(os.listdir("../input/stanford-dogs-dataset/images/Images/{}".format(raza)))
print("Hay {} imagenes".format(numeroTotalDeImagenes))


# # Mostrar x Cantidad de Imagenes de Alguna Raza

# In[ ]:


def mostrarImagenes(raza, numeroDeRazasParaMostrar):
    plt.figure(figsize=(16,16))
    directorioDeImagen = "../input/stanford-dogs-dataset/images/Images/{}/".format(raza)
    imagen = os.listdir(directorioDeImagen)[:numeroDeRazasParaMostrar]
    for i in range(numeroDeRazasParaMostrar):
        img = mpimg.imread(directorioDeImagen + imagen[i])
        plt.subplot(numeroDeRazasParaMostrar/4+1, 4, i+1)
        plt.imshow(img)
        plt.axis('off')
print(razas[10])
mostrarImagenes(razas[10], 20)


# # Cortado de Imagenes
# Para mayo exactitud utilizamos un metodo usado en [https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation](http://) para cortar las imagenes con el fin de que solo se vieran los perros en las imagenes.

# In[ ]:


def cortarImagen(raza, perro, directorioDeData):
    img = Image.open(directorioDeData + 'images/Images/' + raza + '/' + perro + '.jpg')
    tree = ET.parse(directorioDeData + 'annotations/Annotation/' + raza + '/' + perro)
    xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
    xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
    ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
    ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
    img = img.crop((xmin,ymin,xmax,ymax))
    return img


# # Guardamos la Data en Carpetas

# In[ ]:


if 'data' not in os.listdir():
    os.mkdir('data')
os.mkdir('data/perrosCortados/')
os.mkdir('data/perrosNoCortados/')
print('Se ha creado {} carpeta para guardad las nuevas imagenes de perros'.format(len(os.listdir('data'))))


# In[ ]:


directorioDeData = '../input/stanford-dogs-dataset/'
for raza in razas:
    print('guardando imagenes de:' + raza)
    for file in os.listdir(directorioDeData + 'annotations/Annotation/' + raza):
        img = cortarImagen(raza, file, directorioDeData)
        img = img.convert('RGB')
        img.save('data/perrosCortados/' + file + '.jpg')
        img2 = Image.open(directorioDeData + 'images/Images/' + raza + '/' + file + '.jpg')
        img2 = img2.convert('RGB')
        img2.save('data/perrosNoCortados/' + file + '.jpg')


# # Contamos la cantidad de imagenes para verificar.

# In[ ]:


contadorDeImagenes = 0
for image in os.listdir('data/perrosCortados'):
    contadorDeImagenes += 1
print('Numero de imagenes: {}'.format(contadorDeImagenes))
contadorDeImagenes = 0
for image in os.listdir('data/perrosNoCortados'):
    contadorDeImagenes += 1
print('Numero de imagenes: {}'.format(contadorDeImagenes))


# # Mostramos imagenes cortadas para verificar

# In[ ]:


def mostrarImagenesCortadas(numeroDeImagenesCortadas):
    plt.figure(figsize=(10,10))
    directorioDeImagen = "data/perrosCortados/"
    imagen = os.listdir(directorioDeImagen)[:numeroDeImagenesCortadas]
    for i in range(numeroDeImagenesCortadas):
        img = mpimg.imread(directorioDeImagen + imagen[i])
        plt.subplot(numeroDeImagenesCortadas/4+1, 4, i+1)
        plt.imshow(img)
        plt.axis('off')
mostrarImagenesCortadas(15)


# # Procesamos toda la data

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

    for raza in os.listdir('../input/stanford-dogs-dataset/annotations/Annotation/'):
        for perro in os.listdir('../input/stanford-dogs-dataset/annotations/Annotation/' + raza):
            tree = ET.parse('../input/stanford-dogs-dataset/annotations/Annotation/' + raza + '/' + perro)
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
                image = read_image('data/perrosNoCortados/' + perro + '.jpg', bounds)
                images.append(image)
            except:
                print('No image', perro)

    return np.array(images)


x_train = load_images()


# # Generador de Imagenes

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


# # Discriminador

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


# # Modelo GAN

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


PLOT_FRECUENCY = 1
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


# # Entrenamiento

# In[ ]:


def generate_noise(size):
    return np.random.normal(0, 1, size=[size, INPUT_SIZE])


def training(epochs=1, batch_size= 128):
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

    
training(epochs=50)


# # Finalizar

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

