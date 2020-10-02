#!/usr/bin/env python
# coding: utf-8

# ## Latent Walk 
# 
# In the post **[Latent Walk Examples](https://www.kaggle.com/c/generative-dog-images/discussion/98719#latest-569381)** by [Chris Deotte](https://www.kaggle.com/cdeotte) you can find more information about Latent Walks
# 
# 
# ![](http://playagricola.com/Kaggle/exa7519.png)
# 
# <br>
# 
# ### In this kernel I'm going to animate this walk. Like this:
# 
# ![](https://i1.wp.com/media.scoutmagazine.ca/2018/12/giphy-2.gif?fit=480%2C270&ssl=1)
# 
# <br>
# At the end, you'll be able to show your results as a gif :)
# 
# My **baseline** is the code from this amazing kernel: **[Supervised Generative Dog Net](https://www.kaggle.com/cdeotte/supervised-generative-dog-net)**
# 

# In[ ]:


ComputeLB = False
DogsOnly = True

import numpy as np, pandas as pd, os
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt, zipfile 
from PIL import Image 

ROOT = '../input/generative-dog-images/'
if not ComputeLB: ROOT = '../input/'
IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')
breeds = os.listdir(ROOT + 'annotation/Annotation/') 

idxIn = 0; namesIn = []
imagesIn = np.zeros((25000,64,64,3))

# CROP WITH BOUNDING BOXES TO GET DOGS ONLY
if DogsOnly:
    for breed in breeds:
        for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):
            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 
            except: continue           
            tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox') 
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = np.min((xmax - xmin, ymax - ymin))
                img2 = img.crop((xmin, ymin, xmin+w, ymin+w))
                img2 = img2.resize((64,64), Image.ANTIALIAS)
                imagesIn[idxIn,:,:,:] = np.asarray(img2)
                #if idxIn%1000==0: print(idxIn)
                namesIn.append(breed)
                idxIn += 1
                
# RANDOMLY CROP FULL IMAGES
else:
    x = np.random.choice(np.arange(20000),10000)
    for k in range(len(x)):
        img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[x[k]])
        w = img.size[0]; h = img.size[1];
        if (k%2==0)|(k%3==0):
            w2 = 100; h2 = int(h/(w/100))
            a = 18; b = 0          
        else:
            a=0; b=0
            if w<h:
                w2 = 64; h2 = int((64/w)*h)
                b = (h2-64)//2
            else:
                h2 = 64; w2 = int((64/h)*w)
                a = (w2-64)//2
        img = img.resize((w2,h2), Image.ANTIALIAS)
        img = img.crop((0+a, 0+b, 64+a, 64+b))  
        imagesIn[idxIn,:,:,:] = np.asarray(img)
        namesIn.append(IMAGES[x[k]])
        #if idxIn%1000==0: print(idxIn)
        idxIn += 1
    
# DISPLAY CROPPED IMAGES
x = np.random.randint(0,idxIn,25)
for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )
        plt.axis('off')
        if not DogsOnly: plt.title(namesIn[x[k*5+j]],fontsize=11)
        else: plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)
        plt.imshow(img)
    plt.show()


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam


# ### Build Generative Network

# In[ ]:


# BUILD GENERATIVE NETWORK
direct_input = Input((10000,))
x = Dense(2048, activation='elu')(direct_input)
x = Reshape((8,8,32))(x)
x = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# COMPILE
decoder = Model(direct_input, decoded)
decoder.compile(optimizer=Adam(lr=0.005), loss='binary_crossentropy')

# DISPLAY ARCHITECTURE
decoder.summary()


# ### Train Generative Network

# In[ ]:


# TRAINING DATA
idx = np.random.randint(0,idxIn,10000)
train_y = imagesIn[idx,:,:,:]/255.
train_X = np.zeros((10000,10000))
for i in range(10000): train_X[i,i] = 1


# In[ ]:


# TRAIN NETWORK
lr = 0.005
for k in range(50):
    annealer = LearningRateScheduler(lambda x: lr)
    h = decoder.fit(train_X, train_y, epochs = 10, batch_size=256, callbacks=[annealer], verbose=0)
    if k%5==4: print('Epoch',(k+1)*10,'/500 - loss =',h.history['loss'][-1] )
    if h.history['loss'][-1]<0.54: lr = 0.001


# In[ ]:


del train_X, train_y, imagesIn


# # Walking in the Latent Space

# In[ ]:


from PIL import Image, ImageDraw
from IPython.display import Image as IMG 


# In[ ]:


gifs = []
steps = 10
for k in range(5): 
    a = np.random.randint(10000)
    b = np.random.randint(10000)
    print('Walk in Latent Space - IMAGE ',k, ' = gif',k)
    frames = []
    plt.figure(figsize=(20,3))
    for j in range(steps*2):
        xx = np.zeros((10000))
        theta = j%steps/(steps-1)
        if j>=steps: theta = 1-j%steps/(steps-1)
        xx[a] = theta; xx[b] = 1-theta
        xx = xx/(np.sqrt(xx.dot(xx.T)))
        img = decoder.predict(xx.reshape((-1,10000)))
        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
        #img = img.resize((150,150))
        frames.append(img)
        if j < steps: 
            plt.subplot(1,steps,j+1)
            plt.axis('off')
            plt.imshow(img)
    plt.show()
    gifs.append(frames)


# **Generate the gifs**

# In[ ]:


for index,gif in enumerate(gifs):
    frames = gif
    frames[0].save('dogs_'+str(index)+'.gif', format='GIF', append_images=frames[1:], save_all=True, duration=150, loop=0)


# In[ ]:


#IMG(filename='dogs_0.gif', width=200, height = 200)


# ### Image 1
# 
# <img src="dogs_0.gif" height="200" width="200">
# <img src="dogs_1.gif" height="200" width="200">
# 
# ---
# 
# ### Image 2
# 
# <img src="dogs_2.gif" height="200" width="200">
# <img src="dogs_3.gif" height="200" width="200">
# 
# ---
# 
# ### Image 3
# 
# <img src="dogs_4.gif" height="200" width="200">
# <img src="dogs_5.gif" height="200" width="200">
# 
# ---
# 
# ### Image 4
# 
# <img src="dogs_6.gif" height="200" width="200">
# <img src="dogs_7.gif" height="200" width="200">
# 
# ---
# 
# ### Image 5
# 
# <img src="dogs_8.gif" height="200" width="200">
# <img src="dogs_9.gif" height="200" width="200">

# In[ ]:




