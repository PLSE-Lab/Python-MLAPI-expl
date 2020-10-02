#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.model_selection import train_test_split

# Image Processing
import cv2 # OpenCV
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.applications import InceptionResNetV2

# Visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Inline make plot appears on notebook

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# ETC
import os
import gc # Gabage collector


# We have 25,000 images in each class (dog and cat), but we want to try use small amounts of data and use transfer learning later, so we choose 2000 images from each class.

# In[ ]:


train_dir = '../input/train'
test_dir = '../input/test'

train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]
train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]

test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_dogs[:2000] + train_cats[:2000]  # Slice the dataset and use 2000 images from each class
random.shuffle(train_imgs)

# Clear unused data
del train_dogs
del train_cats
gc.collect() # Collect gabage to save memory


# Visualize our training data

# In[ ]:


for ima in train_imgs[0:3]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()


# Resize images using OpenCV (we chose 150x150)

# In[ ]:


nrows = 150
ncols = 150
channels = 3 # 3 for colored images (red, green, blue), 1 for greyscale images


# In[ ]:


# A function to read and process images to the format usable by our model
def read_and_process_image(list_of_images):
    """
        Returns 2 arrays:
            X is an array of resized images
            y is an array of labels
    """
    X = []
    y = []
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncols), interpolation=cv2.INTER_CUBIC)) # Read the image
        # Get the label
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
    return X, y

X, y = read_and_process_image(train_imgs)


# In[ ]:


del train_imgs
gc.collect()


# Lets check the first 5 images. We have to use imshow() because the images are now array of pixels, not raw jpg (so we cannot use matplotlib.image anymore here)

# In[ ]:


plt.figure(figsize=(20, 10))
columns = 5
for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])


# In[ ]:


y[:5]


# In[ ]:


# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)


# In[ ]:


sns.countplot(y)
plt.title("Labels for Cats and Dogs")


# In[ ]:


print("Shape of training data:", X.shape)  # Batch size, Height, Width, Channel
print("Shape of labels:", y.shape)


# Split training data into training data and validation data

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)


# In[ ]:


del X
del y
gc.collect()


# In[ ]:


# Get length of training set and validation set
ntrain = len(X_train)
nval = len(X_val)

# We  will use batch size of 32, batch size should be a factor of 2, 4, 8, 16, 32, 64, ...
batch_size = 32


# Use pretrained model from Keras

# In[ ]:


conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


# Use weights which are trained on imagenet. <br/>
# Assign include_top=False to not to download the fully connected layers (top layers), because I am going to implement it by myself.  
# 
# Lets check the pre-trained model I just loaded.

# In[ ]:


conv_base.summary()


# Use CNN on Keras (Tensorflow backend) and try use small version of an architecture called VGGnet <br>(output filter size of each layer are 32, 64, 128, 512, 1).
# 
# Input shape matched with size of both X.
# 
# (3, 3) is kernel size (that 3x3 pixels size box which scan through image).

# In[ ]:


model = models.Sequential()
model.add(conv_base)  # Add pre-trained model here
model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Use sigmoid activation, because output has only 2 classes


# In[ ]:


model.summary()


# Freeze pre-trained model and only train our own.

# In[ ]:


print('Number of trainable weights BEFORE freezing the conv_base', len(model.trainable_weights))
conv_base.trainable = False
print('Number of trainable weights AFTER freezing the conv_base', len(model.trainable_weights))


# In[ ]:


# Try change learning rate from 0.0001 to 0.0002 (compares to previous model)
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-4), metrics=['acc'])


# In[ ]:


# Use rescaling and image augmentation on training dataset
train_datagen = ImageDataGenerator(rescale=1./255,  # Normalize image to have mean = 0 and standard deviation = 1, helps model learn and update parameters efficiently
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)

# We do not perform augmentation on validation dataset, we only do rescaling
val_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


# Create the image generator
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


# Train for 20 epochs (try reduces it from 64 epochs this time)

# In[ ]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,  
                              epochs=20,                             
                              validation_data=val_generator,
                              validation_steps=nval // batch_size,)


# Plot accuracy and loss

# In[ ]:


acc = history.history['acc']
val_acc  = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# Predicts the first 10 images from test dataset

# In[ ]:


X_test, y_test = read_and_process_image(test_imgs[0:10])  # Outputted y_test will be empty
X_test = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


i = 0
text_labels = []
plt.figure(figsize=(30, 20))
for batch in test_datagen.flow(X_test, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()


# Check confusion matrix and F1 score on validation dataset (should have do it on test dataset, but Kaggle does not have labels for test dataset).

# In[ ]:


i = 0
y_pred = []
for batch in val_datagen.flow(X_val, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        y_pred.append(1)  # Dog
    else:
        y_pred.append(0)  # Cat
    i += 1
    if i == len(X_val):
        break
y_pred = np.array(y_pred)


# In[ ]:


confusion_matrix(y_val, y_pred, labels=None)


# In[ ]:


f1_score(y_val, y_pred, average='weighted') 

