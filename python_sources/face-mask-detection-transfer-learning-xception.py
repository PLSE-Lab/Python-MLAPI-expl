#!/usr/bin/env python
# coding: utf-8

# ## OVERVIEW
# ---
# * Image Processing
# * Transfer Learning With Keras Pretrained Models
# * Bottleneck Feature Extraction
# * Keras library for building a basic Convolutional Neural Network.
# * Predictive Modelling
# * Plotting of Model Performance

# In[ ]:


import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')


import os
from keras.applications import xception
from keras.preprocessing import image
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import cv2
from scipy.stats import uniform

from tqdm import tqdm
from glob import glob


from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Masking
from keras.utils import np_utils, to_categorical
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint



from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ### DATA UTILITIES
# 

# In[ ]:


#copying the pretrained models to the cache directory
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#copy the Xception models
get_ipython().system('cp ../input/keras-pretrained-models/xception* ~/.keras/models/')
#show
get_ipython().system('ls ~/.keras/models')


# In[ ]:


data_folder = '../input/face-mask-detection-data'
categories = ['with_mask', 'without_mask']
len_categories = len(categories)


# ### CREATE A DATAFRAME

# In[ ]:


#show number of images per category
for category in categories:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(data_folder, category)))))


# In[ ]:


train_data = []

for i, category in tqdm(enumerate(categories)):
    class_folder = os.path.join(data_folder, category)    
    for path in os.listdir(os.path.join(class_folder)):
        train_data.append(['{}/{}'.format(category, path), category, i])
df = pd.DataFrame(train_data, columns=['filepath', 'class', 'label'])

#reduce the data
SAMPLE_PER_CATEGORY = 500
df = pd.concat([df[df['class'] == i][:SAMPLE_PER_CATEGORY] for i in categories])

print('DATAFRAME SHAPE: ',df.shape)
df.head()


# ### SHOW SAMPLE IMAGES

# In[ ]:


# function to get an image
def read_img(filepath, size):
    img = image.load_img(os.path.join(data_folder, filepath), target_size=size)
    #convert image to array
    img = image.img_to_array(img)
    return img


# In[ ]:


nb_rows = 3
nb_cols = 5
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 5));
plt.suptitle('SAMPLE IMAGES');
for i in range(0, nb_rows):
    for j in range(0, nb_cols):
        axs[i, j].xaxis.set_ticklabels([]);
        axs[i, j].yaxis.set_ticklabels([]);
        axs[i, j].imshow((read_img(df['filepath'].iloc[np.random.randint(998)], (255,255)))/255);
plt.show();


# ### PREPROCESSING THE IMAGES

# In[ ]:


# function to sharpen the images
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp / 255


# In[ ]:


INPUT_SIZE = 255
X_train = np.zeros((len(df), INPUT_SIZE, INPUT_SIZE, 3), dtype='float')

for i, file in tqdm(enumerate(df['filepath'])):
    img_sharpen = sharpen_image(read_img(file, (255,255)))
    X_train[i] = xception.preprocess_input(np.expand_dims(img_sharpen.copy(), axis=0))
    


# In[ ]:


print('Train Image Shape: ', X_train.shape)
print('Train Image Size: ', X_train.size)


# In[ ]:


#split the data
y = df['label']
train_x, train_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=101)


# ### BOTTLENECK FEATURE EXTRACTION

# In[ ]:


xception_bf = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
bf_train_x = xception_bf.predict(train_x, batch_size=32, verbose=1)
bf_train_val = xception_bf.predict(train_val, batch_size=32, verbose=1)


# In[ ]:


#print shape of feature and size
print('Train Shape: ', bf_train_x.shape)
print('Train Size: ', bf_train_x.size)

print('Validation Shape: ', bf_train_val.shape)
print('Validation Size: ', bf_train_val.size)


# ### MODELLING

# In[ ]:


#optimizer
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

#keras model
model = Sequential()
model.add(Dense(units = 256 , activation = 'relu', input_dim=bf_train_x.shape[1]))
model.add(Dense(units = 64 , activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()


# In[ ]:


#set callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
         ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

#fit the data
history = model.fit(bf_train_x, y_train, batch_size=32, epochs=500, callbacks=callbacks)


# ### LOSS AND ACCURACY

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(14,5))
ax[0].set_title('TRAINING LOSS');
ax[1].set_title('TRAINING ACCURACY');


ax[0].plot(history.history['loss'], color= 'salmon',lw=2);
ax[1].plot(history.history['accuracy'], color= 'green',lw=2);


# In[ ]:


#predict the validation data
predictions = model.predict_classes(bf_train_val)


# ### CLASSIFICATION REPORT

# In[ ]:


print(classification_report(y_val, predictions))


# ### CONFUSION MATRIX

# In[ ]:


con_mat = confusion_matrix(y_val, predictions)

plt.figure(figsize=(8,8))
plt.title('CONFUSION MATRIX')

sns.heatmap(con_mat,
            yticklabels=['with_mask', 'without_mask'], 
            xticklabels=['with_mask', 'without_mask'],
            annot=True, linecolor='black', linewidths=4, square=True);

plt.xlabel('Y_TRUE'), plt.ylabel('PREDICTIONS');

