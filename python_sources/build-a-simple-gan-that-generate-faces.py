#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Get 100k Celebrities Dataset by greg

# ## See some sample Photos

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from random import choice, sample
import cv2
import os
path = '../input/100k/100k'

allp = os.listdir(path)

figure, axis = plt.subplots(5, 5)
image_count = 0
for row in range(5):
  for column in range(5):
    axis[row,column].imshow( cv2.imread( os.path.join(path,choice(allp)) ) )
    axis[row,column].axis('off')
    image_count += 1


# In[ ]:


from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3
INP_DIM = 100

#generates a batch of real faces
def getRealMiniBatch(path='../input/100k/100k',batch_size=32):
  allp = os.listdir(path)
  while True:
    batch = sample(allp,batch_size)
    imgs = []
    for img in batch:
      x = os.path.join(path,img)
      x = img_to_array(load_img( x , target_size=(IMG_WIDTH, IMG_HEIGHT) ))
      imgs.append(x)
    yield (np.array(imgs)/127.5)-1
    
next(getRealMiniBatch()).shape, np.random.normal(0,1,(32,100)).shape


# In[ ]:


from keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation
from keras.layers import Input, Dense, Reshape
from keras.layers import LeakyReLU, Dropout, ZeroPadding2D, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam

def Generator(INP_DIM=100):
    def Block(inp, N, kernel_size=3):
      x = UpSampling2D()(inp)
      x = Conv2D(N, kernel_size=kernel_size, padding='same')(x)
      x = BatchNormalization(momentum=0.8)(x)
      x = Activation('relu')(x)
      return x
    
    inp = Input( shape=(INP_DIM,) )
    x = Dense( 512*4*4, activation='relu' )(inp)
    x = Reshape( (4,4,512) )(x)

    x = Block(x, 512)
    x = Block(x, 512)
    x = Block(x, 256)
    x = Block(x, 256)
    x = Block(x, 128)

    x = Conv2D( 3, kernel_size=3, padding='same' )(x)
    out = Activation('tanh')(x)

    generator = Model(inp, out)
    return generator
  
def Descriminator(width, height, channels):
    def Block(inp, N, kernel_size = 3, strides=2,zeropad=False):
      x = Dropout(0.3)(inp)
      x = Conv2D( N, kernel_size=kernel_size, strides=strides, padding='same' )(x)
      if zeropad:
        x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
      x = BatchNormalization(momentum=0.8)(x)
      x = LeakyReLU(alpha=0.2)(x)
      return x

    inp = Input( shape=(width,height,channels) )
    x = Conv2D( 32, kernel_size=3, strides=2, padding='same' )(inp)
    x = LeakyReLU(alpha=0.2)(x)

    x = Block(x, 64, zeropad=True)
    x = Block(x, 128)
    x = Block(x, 256, strides=1)
    x = Block(x, 512, strides=1)

    x = Dropout(0.25)(x)
    x = Flatten()(x)
    out = Dense(1, activation='sigmoid')(x)

    descriminator = Model(inp,out)
    return descriminator
  
  
def GAN(descriminator,generator,INP_DIM=100):
    #set descriminator untr
    descriminator.trainable=False
    inp = Input( shape=(INP_DIM,) )
    x = generator(inp)
    out = descriminator(x)
    return Model(inp,out)

optimizer1 = Adam(0.00004,0.5)
optimizer2 = Adam(0.00008,0.5)

#How to set descrimator untrainable in GAN
"https://stackoverflow.com/questions/51108076/generative-adversarial-networks-in-keras-doesnt-work-like-expected"

descriminator = Descriminator(width=IMG_WIDTH, height=IMG_HEIGHT, channels=CHANNELS)
descriminator.compile( loss="binary_crossentropy", optimizer=optimizer1, metrics=['accuracy'] )

generator = Generator( INP_DIM=INP_DIM )
gan = GAN(descriminator,generator,INP_DIM=INP_DIM)
gan.compile( loss="binary_crossentropy", optimizer=optimizer2 )


# In[ ]:


#Save 25 generated images for demonstration purposes using matplotlib.pyplot.
def save_figure(epoch,rows,columns,INP_DIM=100):
    noise = np.random.normal(0, 1, (rows * columns, INP_DIM))
    generated_images = generator.predict(noise)
    
    generated_images = generated_images/2 + 0.5
    
    figure, axis = plt.subplots(rows, columns)
    image_count = 0
    for row in range(rows):
        for column in range(columns):
            axis[row,column].imshow(generated_images[image_count, :], cmap='spring')
            axis[row,column].axis('off')
            image_count += 1
    figure.savefig("generated_images/generated_%d.png" % epoch)
    plt.close()


# In[ ]:


BATCH_SIZE = 32
EPOCHS = 500000
START = 1

history = []

def getFakeMiniBatch(batch_size=32, INP_DIM=100):
  while True:
    x = np.random.normal(np.random.normal(0,1,(32,INP_DIM)))
    yield generator.predict(x),x

if not os.path.exists('saved_weights'):
    os.mkdir('saved_weights')
      
real_labels = np.ones((BATCH_SIZE,1))
fake_labels = np.zeros((BATCH_SIZE,1))
real_image_gen = getRealMiniBatch(batch_size=BATCH_SIZE)
fake_image_gen = getFakeMiniBatch(batch_size=BATCH_SIZE, INP_DIM=INP_DIM )

for epoch in range(START,EPOCHS+1):
  fake_images,noise = next(fake_image_gen)
  real_images = next(real_image_gen)
  
  descri_loss_real = descriminator.train_on_batch( real_images, real_labels )
  descri_loss_fake = descriminator.train_on_batch( fake_images, fake_labels )
  descriminator_loss = 0.5*np.add(descri_loss_real,descri_loss_fake)
  
  gan_loss = gan.train_on_batch( noise, real_labels)
  
  history.append( (descriminator_loss, gan_loss) )
  if epoch % 250 == 0:
    print( "{} [ Descriminator loss : {:.2f} acc : {:.2f} ] [ Generator loss : {:.2f} ]".format( 
        epoch, descriminator_loss[0], 100*descriminator_loss[1], gan_loss ) )
    
  if epoch % 500 == 0:
    if not os.path.exists('generated_images'): os.mkdir('generated_images')
    descriminator.save_weights('saved_weights/descriminator_weights.h5')
    generator.save_weights('saved_weights/generator_weights.h5')
    gan.save_weights('saved_weights/gan_weights.h5')
    save_figure(epoch,5,5,INP_DIM)
    print("Figure Saved | All Weights Saved")


# In[ ]:




