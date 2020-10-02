#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

# In[ ]:


import pandas as pd
import os,cv2
from IPython.display import Image
from keras.preprocessing import image
from keras import optimizers
from keras import layers,models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
print(os.listdir("../input"))

import numpy as np


# ## Read csv data

# In[ ]:


input_path = '../input/'
train_path = input_path + 'train/train/'
test_path = input_path + 'test/test/'

train_dir="../input/train/train"
test_dir="../input/test/test"
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/sample_submission.csv')

train_id = train_df['id']
labels = train_df['has_cactus']
test_id = test_df['id']


# ## Split data into training and validation sets

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(train_id, labels, test_size=0.2)


# ## Load images

# In[ ]:


def get_images(ids, filepath):
    arr = []
    for img_id in ids:
        img = plt.imread(filepath + img_id)
        arr.append(img)
    
    arr = np.array(arr).astype('float32')
    arr = arr / 255
    return arr


# In[ ]:


x_train = get_images(ids=x_train, filepath=train_path)
x_val = get_images(ids=x_val, filepath=train_path)
test = get_images(ids=test_id, filepath=test_path)

img_dim = x_train.shape[1:]


# ## Plot data

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=3)
ax = ax.ravel()
plt.tight_layout(pad=0.2, h_pad=2)

for i in range(6):
    ax[i].imshow(x_train[i])
    ax[i].set_title('has_cactus = {}'.format(y_train.iloc[i]))


# ## Define batch size, epochs and steps

# In[ ]:


batch_size = 64
epochs = 30
steps = x_train.shape[0] // batch_size


# ## Define model

# In[ ]:


inputs = Input(shape=img_dim)

densenet121 = DenseNet121(weights='imagenet', include_top=False)(inputs)

flat1 = Flatten()(densenet121)
dense1 = Dense(units=256, use_bias=True)(flat1)
batchnorm1 = BatchNormalization()(dense1)
act1 = Activation(activation='relu')(batchnorm1)
drop1 = Dropout(rate=0.5)(act1)

out = Dense(units=1, activation='sigmoid')(drop1)

model = Model(inputs=inputs, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy')


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=2, mode='max')

img_aug = ImageDataGenerator(rotation_range=20, vertical_flip=True, horizontal_flip=True)
img_aug.fit(x_train)

model.fit_generator(img_aug.flow(x_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=steps, epochs=epochs, 
                    validation_data=(x_val, y_val), callbacks=[reduce_lr], 
                    verbose=2)


# ## Get predictions

# In[ ]:


test_pred = model.predict(test, verbose=2)


# ## Create submission file

# In[ ]:


test_df['has_cactus'] = test_pred
test_df.to_csv('submission.csv', index=False)

