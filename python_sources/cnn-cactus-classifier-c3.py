#!/usr/bin/env python
# coding: utf-8

# ### Load Data
# 
# Before we can actually train our model, we need to get the data. The data used here is from the [Aerial Cactus Identification](https://www.kaggle.com/c/aerial-cactus-identification) challenge on Kaggle.

# In[ ]:


import pandas as pd
import cv2
import os

train_csv = pd.read_csv('../input/train.csv')

def load_imgs(path):
    imgs = {}
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        imgs[f] = cv2.imread(fname)
    return imgs

img_train = load_imgs('../input/train/train/')
img_test = load_imgs('../input/test/test/')


# ### Process Data
# 
# I also need to put the data into a usable form. I do this below.

# In[ ]:


import numpy as np

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    X_train.append(img_train[row['id']])
    Y_train.append(int(row['has_cactus']))

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array([img_test[f] for f in img_test])

print('Training data shape:', X_train.shape, '=>', Y_train.shape)


# In[ ]:


# Validation split
idxs = np.random.permutation(17500)

x_train = X_train[idxs[:12000]]
y_train = Y_train[idxs[:12000]]

x_valid = X_train[idxs[12000:]]
y_valid = Y_train[idxs[12000:]]


# ### Build Model
# 
# Next, I will create my model. In this demo, I am using CNN because it is robust for computer vision problems.

# In[ ]:


from keras.layers import *
from keras.models import Sequential

def add_conv_level(model, **args):
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', **args))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

def make_model():
    model = Sequential()
    
    for i in range(3):
        if i:
            add_conv_level(model)
        else:
            add_conv_level(model, input_shape=(32,32,3))
    
    model.add(Flatten())
    
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model
    
model = make_model()
model.summary()


# In[ ]:


from keras.optimizers import *

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# ### Train The Model
# 
# Now that everything has been created, I can perform training on the data. I use the validation data to gauge how well my model is performing.

# In[ ]:


model.fit(x_train, y_train,
          epochs=16, batch_size=32,
          validation_data = (x_valid, y_valid))


# Once I am satisfied with my choice of hyperparameters, I can go ahead and train the model on the entire dataset. I will remake, recompile, and retrain my model.

# In[ ]:


model = make_model()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=16, batch_size=32)

# Save the model for future usage
model.save('model.h5')


# ### Evaluate The Model
# 
# With a "perfected" model, I am free to make predictions and submit them to the competition.

# In[ ]:


pred = model.predict(X_test)


df = pd.DataFrame({
    'id': [f for f in img_test],
    'has_cactus': [int(x[0] >= 0.5) for x in pred]
})

print(df)


# In[ ]:


df.to_csv('predictions.csv', index=False)

