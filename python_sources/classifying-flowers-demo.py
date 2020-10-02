#!/usr/bin/env python
# coding: utf-8

# # Final Project: Classifying Flowers
# It's nearing the end of year and it's time we work on one final project. First we learned about AI, and now we are going to combine it with web scraping. The first thing we are going to do is create a neural network to classify the flowers. Then I will direct you to a website where you will have to scrape all the images off and classify them. Whover does it with the best accuracy will win. I will show you a benchmark for this competition. We are going to use Keras along with some other libraries we will need to extract data from the files. Here is the stuff we will be doing in this project:
# 1. Opening and Pre-processing Image Files
# 2. Data Augmentation
# 3. Using Pretrained Models
# 4. Hyperparameter Search
# 5. Saving a Neural Network
# 6. Web Scraping
# I am trying to give you a more realistic view of what a real data scientist will do but on a short time, so follow along!

# In[1]:


import numpy as np
from PIL import Image
from keras.applications import *
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
import os


# ## Opening Image Files
# This is a very straightforward step. We can automatically extract images from an archived folder. Let's see what kinds of flowers are in this dataset:

# In[2]:


flowers = os.listdir('../input/flowers-recognition/flowers/flowers')
print (flowers)


# Ok. So we have 5 classes: Rose, Dandelion, Sunflower, Tulip, and Daisy. Each folder has images of the respective class in it. If you read the description of the dataset, the images' dimensions are not all the same! We will have to resize them! Since I will be using a pretrained model and fine-tuning it, I will have to resize them to a specific size, regardless of the dimensions they were before. I will resize them to the 224x224. We will use `os.listdir('..input/flowers/flowers/flower_name')` for each folder so we can get their path. Then we can open the images!
# <br>
# Here is how we are going to do this: We will create a dictionary, where each key is a label. Then we will create a list corresponding to each key. We will go over all the files and add their paths to their respective label.

# In[3]:


paths = {'rose': [],
         'dandelion': [],
         'sunflower': [],
         'tulip': [],
         'daisy': []
        }

for key, images in paths.items():
    for filename in os.listdir('../input/flowers-recognition/flowers/flowers/'+key):
        paths[key].append('../input/flowers-recognition/flowers/flowers/'+key+'/'+filename)
    
    print (len(images),key,'images')


# In[4]:


X = []
Y = []
mapping = {'rose': 0,
         'dandelion': 1,
         'sunflower': 2,
         'tulip': 3,
         'daisy': 4
        }
for label,image_paths in paths.items():
    for path in image_paths:
        if '.py' not in path:
            image = Image.open(path)
            image = image.resize((224,224))
            X.append(np.array(image))

            one_hot = np.array([0.,0.,0.,0.,0.])
            one_hot[mapping[label]] = 1.
            Y.append(one_hot)


# In[5]:


aug_X = []
aug_Y = []

for image in X:
    aug_X.append(np.flip(image,1))

aug_Y = Y


# In[6]:


X = X + aug_X
Y = Y + aug_Y


# In[7]:


len(X)


# In[8]:


from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras import backend as K


# In[9]:


base_model = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))


# In[10]:


base_model.load_weights('../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[11]:


for layer in base_model.layers:
    layer.trainable = False


# In[12]:


output = base_model.output


# In[13]:


from keras.layers import Flatten


# In[14]:


output = Flatten()(output)


# In[15]:


output = Dense(5, activation='softmax')(output)


# In[16]:


model = Model(inputs=base_model.input, outputs=output)


# In[17]:


model.summary()


# In[18]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[19]:


model.fit(np.stack(X,axis=0),np.stack(Y,axis=0),validation_split=0.1,batch_size=8,epochs=15,verbose=1)


# In[ ]:




