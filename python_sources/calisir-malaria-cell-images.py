#!/usr/bin/env python
# coding: utf-8

# ## MALARIA CELL IMAGES
# What we will do:
# Download the data from Kaggle, <br>
# Prepare the data, <br>
# Train at least with:
# * Convolutional Neural Network,
# * Use Transfer Learning to build other model(s) on top of neural networks trained on ImageNet. For ready-to-be-used neural networks in Keras, please see https://keras.io/applications/

# ### Importing Dependencies
# 

# In[ ]:


import numpy as np
np.random.seed(1000)
import pandas as pd
import matplotlib.pyplot as plt
import os,cv2
from PIL import Image
from IPython.display import SVG
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm,tqdm_notebook
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.mobilenet import preprocess_input
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.densenet import DenseNet121
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D,GlobalMaxPooling2D,Conv2D, MaxPooling2D,BatchNormalization


import keras
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential,Input,Model


# # With CNN

# ## Data Preperation

# In[ ]:


DATA_DIR = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/'
SIZE = 64
dataset = []
label = []


# In[ ]:


import os
print(os.listdir("../input/mobilenet")) # Imported From Kaggle


# In[ ]:


parasitized_images = os.listdir(DATA_DIR + 'Parasitized/')
uninfected_images = os.listdir(DATA_DIR + 'Uninfected/')
X_train1= []
X_test1=[]
for i, image_name in enumerate(parasitized_images):
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Parasitized/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            X_train1.append(image)
            dataset.append(np.array(image))
            label.append(0)
            
for i, image_name in enumerate(uninfected_images):
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Uninfected/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            X_test1.append(image)
            dataset.append(np.array(image))
            label.append(1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 0)


# ## Visualization

# ### Infected Images

# In[ ]:


plt.figure(figsize = (20, 12))
for index, image_index in enumerate(np.random.randint(len(parasitized_images), size = 10)):
    plt.subplot(1, 10, index+1)
    plt.imshow(dataset[image_index])


# ### Uninfected Images

# In[ ]:


plt.figure(figsize = (20, 12))
for index, image_index in enumerate(np.random.randint(len(uninfected_images), size = 10)):
    plt.subplot(1, 10, index+1)
    plt.imshow(dataset[len(parasitized_images) + image_index])


# ## Model Preperation

# In[ ]:


classifier = None
classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
classifier.add(BatchNormalization(axis = -1))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
classifier.add(BatchNormalization(axis = -1))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(activation = 'relu', units=512))
classifier.add(BatchNormalization(axis = -1))
classifier.add(Dropout(0.2))
classifier.add(Dense(activation = 'relu', units=256))
classifier.add(BatchNormalization(axis = -1))
classifier.add(Dropout(0.2))
classifier.add(Dense(activation = 'sigmoid', units=2))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(classifier.summary())


# In[ ]:


train_generator = ImageDataGenerator(rescale = 1/255,
                                     zoom_range = 0.3,
                                     horizontal_flip = True,
                                     rotation_range = 30)

test_generator = ImageDataGenerator(rescale = 1/255)

train_generator = train_generator.flow(np.array(X_train),
                                       y_train,
                                       batch_size = 64,
                                       shuffle = False)

test_generator = test_generator.flow(np.array(X_test),
                                     y_test,
                                     batch_size = 64,
                                     shuffle = False)


# ## Model Training

# In[ ]:


train_generator = ImageDataGenerator(rescale = 1/255,
                                     zoom_range = 0.3,
                                     horizontal_flip = True,
                                     rotation_range = 30)

test_generator = ImageDataGenerator(rescale = 1/255)

train_generator = train_generator.flow(np.array(X_train),
                                       y_train,
                                       batch_size = 64,
                                       shuffle = False)

test_generator = test_generator.flow(np.array(X_test),
                                     y_test,
                                     batch_size = 64,
                                     shuffle = False)


# In[ ]:


history = classifier.fit_generator(train_generator,
                                   steps_per_epoch = len(X_train)/64,
                                   epochs = 5,
                                   shuffle = False)


# In[ ]:


print("Test_Accuracy(after augmentation): {:.2f}%".format(classifier.evaluate_generator(test_generator, steps = len(X_test), verbose = 1)[1]*100))


# # With ImageNet Model

# In[ ]:


import os
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/")) # Imported From Kaggle


# In[ ]:


from keras.applications import MobileNet
mobilenet=MobileNet(weights='../input/mobilenet/mobilenet_1_0_224_tf.h5') 


# In[ ]:


base_model=MobileNet(weights='../input/mobilenetnotop/mobilenet_1_0_224_tf_no_top.h5', include_top=False) 
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(2,activation='softmax')(x)


# ## Model Preperation

# In[ ]:


from keras.models import Model

model=Model(inputs=base_model.input,outputs=preds)
#model.summary()
for layer in model.layers[:-5]:
    layer.trainable=False


# In[ ]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory(
    '../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


# ## Model Training

# In[ ]:


model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n/train_generator.batch_size,
    epochs=5
)

