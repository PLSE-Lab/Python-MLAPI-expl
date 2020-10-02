#!/usr/bin/env python
# coding: utf-8

# # Hello everyone

# This is my first attempt to build a CNN from scratch. Untill now I am just doing models from 'from keras.dataset import ...', so I hope it works fine (and works as you see at the end and I am very happy with the results)
# 
# I will describe as clearly as possible for you if you're a beginner and for myself because I probably gonna use this notebook as my future guides as well. So I want you to understand clearly what I am doing
# 
# At the end of the page I'm gonna put some notes just for justify some stuffs I know that can be written in another way or why I'm importing x library since y library can do the same.
# 
# So lets begin ...

# In[ ]:


import keras
from keras.preprocessing import image
from glob import glob
import cv2, os
import numpy as np
import matplotlib.pyplot as plt


# Note that even it is a jupyter notebook file I not: using %matplotlib inline
#     
# The reason is just because it is easier to run if I download it as .py
# 
# I am using the kaggle's notebook path for the train folder and we don't need the test for this project. We gonna create and train our network for those images and thats all.
# 
# I write ROW and COL in uppercase because this is the way I learn and used to write my codes with Keras so it is a way to differentiate things that are more important or belong to Keras parte from the most part of code but in this case, we need these variables to resize our image.

# In[ ]:


path = '../input/train/'

ROW, COL = 96, 96


# In[ ]:


dogs, cats = [], []
y_dogs, y_cats = [], []


# # Data pre processing

# Ok. I am like to separate Cats from Dogs (because they tend to fight each other - just kidding). To be honest I don't think you need to do this but I think it is a better way to understand what I am doing and best if you someday need to use this way for a further projects.
# 
# y_dogs and y_cats are my labels.
# 
# The code bellow are just a definition and selfexplanatory but lets see
# 
# I am getting all 'dog' images. In this case there are just .jpg in this folder so it is unnecessary to write things like 'dog*.jpg' and for each dog image (dog_img) I am creating a variable for save all transformations I am going to do.
# 
# Read an image then change its color to gray because there are more variables that differentiate a dog from a cat. Resize this image for an OK size (I've tested it and 50x50px still nice but lets use 96x96px) and then convert it into array. If you don't convert it's ok but to read and proccess images as images itself is slower than to proccess them as arrays

# In[ ]:


def load_dogs():
    print('Loading all dog images\n')
    dog_path = os.path.join(path, 'dog*')
    for dog_img in glob(dog_path):
        dog = cv2.imread(dog_img)
        dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
        dog = cv2.resize(dog, (ROW, COL))
        dog = image.img_to_array(dog)
        dogs.append(dog)
    print('All dog images loaded')


# In[ ]:


def load_cats():
    print('Loading all cat images\n')
    cat_path = os.path.join(path, 'cat*')
    for cat_img in glob(cat_path):
        cat = cv2.imread(cat_img)
        cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
        cat = cv2.resize(cat, (ROW, COL))
        cat = image.img_to_array(cat)
        cats.append(cat)
    print('All cat images loaded')


# In[ ]:


load_dogs()


# In[ ]:


print('#################################')


# In[ ]:


load_cats()


# In[ ]:


print('Lenght of our dogs array: {}\nLenght of our cats array: {}'.format(len(dogs),len(cats)))


# Everything is ok. We need exactly 12,500 for cats and 12,500 for dogs

# In[ ]:


classes = ['dog', 'cat']


# Acctually it is another step that you can skip (with propper changes with code bellow) but to make it more clearly I am gonna make it explicity that our classes are 'dog' and 'cat'.
# 
# And bellow I am just want to see if our code store everything correctly. You know. Sometimes it is easier to work with arrays but are you sure that those numbers means what you whant to they supposed to mean?

# In[ ]:


import random


# In[ ]:


plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    img = image.array_to_img(random.choice(dogs))
    plt.imshow(img)
    
    plt.axis('off')
    plt.title('Suposed to be a {}'.format(classes[0]))
    
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    img = image.array_to_img(random.choice(cats))
    plt.imshow(img)
    
    plt.axis('off')
    plt.title('Suposed to be a {}'.format(classes[1]))
    
plt.show()


# Nice. They mean what I want to.
# 
# But my computer doesn't understand what dogs and cats are but it knew very well what a 0 and 1 means.
# 
# Lets give these labels 1 for dogs and 0 for cats, as Kaggle inform.

# In[ ]:


y_dogs = [1 for item in enumerate(dogs)]
y_cats = [0 for item in enumerate(cats)]


# In[ ]:


print('Len of dogs labels: {}\nLen of cats labels: {}'.format(len(y_dogs), len(y_cats)))


# # Almost done for data preparation

# What I am going to do is to write an X file for our trainning just like the convetion. Put all my values into one array and them I am going to normalize my X just to fit values between 0 and 1 and for this reason I need to specify that my X is a float type
# 
# For my y vales or my labels I am just writing them as one array also because we need to use like this to make what is called: One-hot encoding.

# In[ ]:


dogs = np.asarray(dogs).astype('float32')
cats = np.asarray(cats).astype('float32')
y_dogs = np.asarray(y_dogs).astype('int32')
y_cats = np.asarray(y_cats).astype('int32')
dogs /= 255
cats /= 255


# In[ ]:


X = np.concatenate((dogs,cats), axis=0)
y = np.concatenate((y_dogs, y_cats), axis=0)


# # Start our CNN model

# In[ ]:


from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import save_model, load_model


# Now I am just setting my network from what is called Convolution Neural Network. Those variables are not arbitrary and you can change for and from your necessity but the composition for being a CNN still remmaing:
# 
# Convolutional -> Convolutional -> MaxPooling -> Convolutional -> Convolutional -> MaxPooling
# 
# As I said those variables are not arbitrary but I am happy with my results. I'm not gonna explain everything. The harder part is over with our data pre processing.
# 
# Now I suggest you to learn more about networks because it worth. Even myself still very beginner.
# 
# We gonna run this network for 100 epochs and I think it is enought to get a great result.
# 
# One more thing. I am changing my y again, passing my labels to a matrice that can be 0 or 1, deppending on what my model learn and for this I said that I have 2 classes getting from the lenght of my classes variable which in my opinion is a great way to inform to your model how manny classes you have.

# In[ ]:


IMG_CHANNEL = 1
BATCH_SIZE = 128
N_EPOCH = 100
VERBOSE = 2
VALIDAION_SPLIT = .2
OPTIM = Adam()
N_CLASSES = len(classes)


# In[ ]:


y = np_utils.to_categorical(y, N_CLASSES)
print('One-Hot Encoding done')


# In[ ]:


model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(ROW, COL, IMG_CHANNEL), activation='relu'),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(.25),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(.5),
    Dense(N_CLASSES, activation='softmax')
])


# In[ ]:


print('The model was created by following config:')
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])


# In[ ]:


checkpoint = ModelCheckpoint('model_checkpoint/dogs_vs_cats_redux_checkpoint.h5')


# Everything setted up so ...

# # Lets run our model

# In[ ]:


print('#################################')
print('########### RUNNING #############')
model.fit(X, y, batch_size=BATCH_SIZE, epochs=N_EPOCH, validation_split=VALIDAION_SPLIT, verbose=VERBOSE)


# In[ ]:


print('############ SCORE ##############')
scores = model.evaluate(X, y, verbose=2)
print('MODEL ACCURACY\n{}: {}%'.format(model.metrics_names[1], scores[1]*100))


# That is it.
# 
# Everything I done here was based on what I've read and learn from the book: Deep Learning with Keras by Antonio Gulli and Sujit Pal
# 
# And material collected from internet, specially sentdex and rmotr
# 
# Thank you
