#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from matplotlib import pyplot


# In[ ]:


import os
import numpy as np
import glob
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt, zipfile


# In[ ]:


new_size = 96
rel_size = 6


# In[ ]:


# define the standalone discriminator model
def define_discriminator(in_shape=[new_size,new_size,3]):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
   
    '''
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[112, 112, 3]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    '''
    
    # compile model
    opt = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


# In[ ]:


# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 8x8 image
    n_nodes = 256 * rel_size * rel_size
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((rel_size, rel_size, 256)))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 64x64
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 128x128
    model.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    
    '''
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 112, 112, 128)    
    
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    '''
    
    model.summary()
    return model


# In[ ]:


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    model.summary()
    return model


# In[ ]:


root_images="../input/all-dogs/all-dogs/"
root_annots="../input/annotation/Annotation/"

all_images=os.listdir(root_images)
print(f"Total images : {len(all_images)}")

breeds = glob.glob('../input/annotation/Annotation/*')
annotation=[]
for b in breeds:
    annotation+=glob.glob(b+"/*")
print(f"Total annotation : {len(annotation)}")

breed_map={}
for annot in annotation:
    breed=annot.split("/")[-2]
    index=breed.split("-")[0]
    breed_map.setdefault(index,breed)
    
print(f"Total Breeds : {len(breed_map)}")


# In[ ]:


def bounding_box(image):
    bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])
    tree = et.parse(bpath)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox') # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
    return (xmin,ymin,xmax,ymax)


# In[ ]:


plt.figure(figsize=(15,15))
for i,image in enumerate(all_images):
    bbox=bounding_box(image)
    im=Image.open(os.path.join(root_images,image))
    im=im.crop(bbox)
    
    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(im)    
    if(i==15):
        break


# In[ ]:


train_images = []
for i,image in enumerate(all_images):
    bbox=bounding_box(image)
    #im=Image.open(os.path.join(root_images,image)).convert('L', (0.2989, 0.5870, 0.1140, 0)) # I was playing around with b&w pictures but it is actually not that hard to do it in RGB
    im=Image.open(os.path.join(root_images,image))
    im=im.crop(bbox)
    
    #new_size  = 128
    if (im.size[0] > new_size and im.size[0] > new_size):
        if ( im.size[0] <= im.size[1] ):
            new_height = new_size * im.size[1] / im.size[0]
            new_width = new_size
        else:
            new_width = new_size * im.size[0] / im.size[1]
            new_height = new_size
            
        im = im.resize((int(new_width), int(new_height)), Image.ANTIALIAS)
        im = im.crop((0,0,new_size,new_size))
        #im = np.asarray(im)
        train_images.append(im)


# In[ ]:


import random

plt.figure(figsize=(15,15))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(train_images[random.randint(1,len(train_images)-1)])    


# In[ ]:


num_images = len(train_images)
print(num_images)

train_data = []
for i in range(num_images):
    train_data.append(np.asarray(train_images[i]))
    
for i in range(num_images):
    train_data[i] = np.reshape(train_data[i], (new_size, new_size, 3)).astype('float32')
    train_data[i] = (train_data[i] - 127.5) / 127.5 # Normalize the images to [-1, 1]
    
dataset = np.array(train_data)    
    
del train_images
del train_data


# In[ ]:


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# In[ ]:


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# In[ ]:


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


# In[ ]:


# create and save a plot of generated images
def save_plot(examples, epoch, n=4):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    fig = plt.figure(figsize=(17,17))
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    plt.show()
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


# In[ ]:


# create and save a plot of generated images
def show_plot(g_model, latent_dim, n=4):
    # scale from [-1,1] to [0,1]
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, 100)
    x_fake = (x_fake + 1) / 2.0
    fig = plt.figure(figsize=(17,17))
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(x_fake[i])
    # save plot to file
    plt.show()
    pyplot.close()


# In[ ]:


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)


# In[ ]:


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            # print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        print(f"epoch {i+1} done")
        if (i+1) % 5 == 0:
            print(f"exemplary results for epoch {i+1}")
            show_plot(g_model, latent_dim)
        # evaluate the model performance, sometimes
        if (i+1) % 50 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
            


# In[ ]:


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)


# In[ ]:


# train model
train(g_model, d_model, gan_model, dataset, latent_dim, 600)


# In[ ]:


z = zipfile.PyZipFile('images.zip', mode='w')
for k in range(10000):
    f = str(k)+'.png'
    
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, 1)
    x_fake = (x_fake * 127.5 + 127.5).astype(np.uint8)
    im = np.asarray(x_fake)
    
    dogPicture = Image.fromarray(im[0,:,:,:])
    dogPicture = dogPicture.resize((64,64))
    dogPicture.save(f,'PNG') 
    z.write(f) 
    os.remove(f)
    
    if k % 1000==0: 
        print(k)
z.close()

