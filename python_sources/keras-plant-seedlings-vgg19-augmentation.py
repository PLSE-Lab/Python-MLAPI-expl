#!/usr/bin/env python
# coding: utf-8

# ## Keras VGG19 + Data Augmentation + Transfer Learning, Kaggle Plant Seedlings Classification 
# 
# Simple Keras implementation with Transfer Learning. 
# 
# * You must run this on a GPU.
# <br>
# 
# [MY GITHUB](https://github.com/AtriSaxena/)
# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import cv2
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


# In[ ]:


CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)


# In[ ]:


SEED = 1987
data_dir = '../input/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))


# ### Number of training images for each Category

# In[ ]:


for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))


# In[ ]:


train = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir, category)):
        train.append(['train/{}/{}'.format(category, file), category_id, category])
train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
train.head(2)
train.shape


# In[ ]:


test = []
for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
test = pd.DataFrame(test, columns=['filepath', 'file'])
test.head(2)
test.shape


# ### See some of the Images

# In[ ]:


fig = plt.figure(1, figsize=(NUM_CATEGORIES, NUM_CATEGORIES))
grid = ImageGrid(fig, 111, nrows_ncols=(NUM_CATEGORIES, NUM_CATEGORIES), axes_pad=0.05)
i = 0
for category_id, category in enumerate(CATEGORIES):
    for filepath in train[train['category'] == category]['file'].values[:NUM_CATEGORIES]:
        ax = grid[i]
        img = Image.open("../input/"+filepath)
        img = img.resize((240,240))
        ax.imshow(img)
        ax.axis('off')
        if i % NUM_CATEGORIES == NUM_CATEGORIES - 1:
            ax.text(250, 112, filepath.split('/')[1], verticalalignment='center')
        i += 1
plt.show();


# ## Model Preparation

# In[ ]:


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (240, 240, 3))


# ### Freezing first few layers
# 
# freeze the first few layers as these layers will be detecting edges and blobs,

# In[ ]:


for layer in model.layers[:5]:
    layer.trainable = False


# ### Adding output Layer

# In[ ]:


x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(12, activation="softmax")(x) 


# In[ ]:


model_final = Model(input = model.input, output = predictions)
#compling our model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# In[ ]:


model_final.summary() #Model summary


# ## Data Augmentation

# In[ ]:


gen = ImageDataGenerator(
            rotation_range=360.,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True)


# In[ ]:


train_data_dir = "../input/train"
train_generator = gen.flow_from_directory(
                        train_data_dir,
                        target_size = (240, 240),
                        batch_size = 16, 
                        class_mode = "categorical")


# In[ ]:


checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')


# ### Train our Model

# In[ ]:


model_final.fit_generator(
                    train_generator,
                    epochs = 50,
                    shuffle= True,
                    callbacks = [checkpoint, early])


# ## Predicting the test images from trained model

# In[ ]:


classes = train_generator.class_indices  
print(classes)


# In[ ]:


#Invert Mapping
classes = {v: k for k, v in classes.items()}
print(classes)


# ### Prediction on each image

# In[ ]:


prediction = []
for filepath in test['filepath']:
    img = cv2.imread(os.path.join(data_dir,filepath))
    img = cv2.resize(img,(240,240))
    img = np.asarray(img)
    img = img.reshape(1,240,240,3)
    pred = model_final.predict(img)
    prediction.append(classes.get(pred.argmax(axis=-1)[0])) #Invert Mapping helps to map Label


# In[ ]:


test = test.drop(columns =['filepath']) #Remove file path from test DF


# In[ ]:


sample_submission.head()


# In[ ]:


pred = pd.DataFrame({'species': prediction})
test =test.join(pred)


# ### Final submission File

# In[ ]:


test.to_csv('submission.csv', index=False)


# In[ ]:


test.head()

