#!/usr/bin/env python
# coding: utf-8

# ## Intro
# 
# Health is always a big topic, I really like the idea that Computers can help to make our lives healthier or can draw your attention more towards it.
# 
# Making a Machine Learning model which can recognize that our mole is bengin or malignant only from an image, makes me think that AI will make unimaginable changes in our lifes in the following decades or years.
# 
# Maybe this kernel what you read is the part of this change. _Feel free to read, use or learn from it!_

# # Skin Cancer: Malignant vs Benign
# This dataset contains a balanced dataset of images of benign skin moles and malignant skin moles.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import gc
from PIL import Image

import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
from keras.applications import VGG16
from tensorflow import set_random_seed
from sklearn.utils import check_random_state


sns.set()
np.random.seed(0);
set_random_seed(0);
check_random_state(0);


# ## Content
# 
# The data consists of two folders with each 1800 pictures (224x244) of the two types of moles.

# In[ ]:


# Save the file path of each image and separate them to different classes
#
# Labels:
# 0 -> benign
# 1 -> malignant

train_imgs, test_imgs = [], []
train_labels, test_labels = [], []

for img_path in os.listdir('../input/data/train/benign'):
    train_imgs.append('../input/data/train/benign/' + img_path)
    train_labels.append(0)
    
for img_path in os.listdir('../input/data/train/malignant'):
    train_imgs.append('../input/data/train/malignant/' + img_path)
    train_labels.append(1)
    
for img_path in os.listdir('../input/data/test/benign'):
    test_imgs.append('../input/data/test/benign/' + img_path)
    test_labels.append(0)
    
for img_path in os.listdir('../input/data/test/malignant'):
    test_imgs.append('../input/data/test/malignant/' + img_path)
    test_labels.append(1)
    
train_imgs, test_imgs = np.array(train_imgs), np.array(test_imgs)
train_labels, test_labels = np.array(train_labels), np.array(test_labels)
    
class_distribution = np.bincount(np.concatenate([train_labels, test_labels]))
    
print('Size of train set:', len(train_imgs))
print('Size of test set:', len(test_imgs))
print(class_distribution[0], 'benign labeled samples and', class_distribution[1], 'malignant')


# ## Data selection
# 
# Because of the size of the dataset I merge the train and the test sets to slice only 10% of the whole training data as the test set.
# 
# Then I slice 7% of the train set to use it for validation.

# In[ ]:


# Load the images to memory
xtrain, xtest = [], []
ytrain, ytest = train_labels, test_labels

for filename in tqdm.tqdm(train_imgs):
    xtrain.append(np.array(Image.open(filename)))
    
for filename in tqdm.tqdm(test_imgs):
    xtest.append(np.array(Image.open(filename)))
    
del train_imgs, test_imgs, train_labels, test_labels
xtrain, xtest = np.array(xtrain), np.array(xtest)

# Merge and split train and test set to have more train data
data = np.concatenate([xtrain, xtest])
labels = np.concatenate([ytrain, ytest])

# Spliting data to train, validation and test values
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=.1, random_state=0)
xtra, xval, ytra, yval = train_test_split(xtrain, ytrain, test_size=.05, random_state=0, shuffle=False)

gc.collect()
print('Shape of the new train set:', xtra.shape)
print('Shape of the new test set:', xtest.shape)
print('Shape of the validation set:', xval.shape)


# Now I generate more training images by changing some of them.

# In[ ]:


data_generator = ImageDataGenerator(rotation_range=90,
                                    width_shift_range=0.15,
                                    height_shift_range=0.15,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    brightness_range=[0.8, 1.1],
                                    fill_mode='nearest')

new_samples, new_labels = next(data_generator.flow(xtra, ytra, batch_size=len(xtra)))
xtra = np.concatenate([xtra, new_samples])
ytra = np.concatenate([ytra, new_labels])

del new_samples, new_labels
print('New number of training samples:', len(xtra))


# Use the simple normalization technique where we just divide each pixel value by `255`.

# In[ ]:


# Normalizing values
xtra = xtra.astype('float32') / 255.
xtest = xtest.astype('float32') / 255.
xval = xval.astype('float32') / 255.

print('Training data shape:', xtra.shape)
print('Min value:', xtra.min())
print('Max value:', xtra.max())


# ## Building and training the model
# 
# After this I build a CNN model in keras.

# In[ ]:


"""
# the commented model
# had 0.83 test accuracy

model.add(Conv2D(32, (3, 3,), activation='relu', input_shape=(224, 224, 3,)))
model.add(Conv2D(32, (3, 3,), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2,)))
model.add(Dropout(.25))
model.add(Conv2D(64, (3, 3,), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3,), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2,)))
model.add(Dropout(.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))
"""

# Build the model
model = Sequential()

model.add(VGG16(include_top=False, input_shape=(224, 224, 3,)))
model.add(Flatten())
model.add(Dense(32))
model.add(LeakyReLU(0.001))
model.add(Dense(16))
model.add(LeakyReLU(0.001))
model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable = False

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


# Now we can train the model.

# In[ ]:


# Train the model
N_EPOCHS = 20
h = model.fit(xtra, ytra, validation_data=(xval, yval), epochs=N_EPOCHS, batch_size=64)


# Visualizing the accuracy and losses to check if the model is overfit.

# In[ ]:


# Plotting accuracy history
plt.figure(figsize=(15, 8))
plt.scatter(range(N_EPOCHS), h.history['acc'], marker='x', label='Training accuracy');
plt.plot(range(N_EPOCHS), h.history['val_acc'], color='green', label='Validation accuracy');
plt.legend();
plt.title('Accuracy');

# Plotting loss history
plt.figure(figsize=(15, 8))
plt.scatter(range(N_EPOCHS), h.history['loss'], marker='x', label='Training loss');
plt.plot(range(N_EPOCHS), h.history['val_loss'], color='green', label='Validation loss');
plt.legend();
plt.title('Loss');


# Evaluate the model on test data to double check it's accuracy, and save it to this kernel.

# In[ ]:


print('Accuracy on test set:', model.evaluate(xtest, ytest)[1])
model.save('model.h5')


# ### Acknowledgements
# 
# All the rights of the Data are bound to the ISIC-Archive rights (https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main). I do not take any responsibility for the right-infringement of any kernels. Thus, do not monetize this any of your models done on this data :).
# 
# ## Inspiration
# 
# I hope that with this dataset, some machine learning model might achieve better and cheaper prediction than a dermatologist. Have fun!
# 
# _The second the third and the last markdown cells are from this dataset's [description](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign)_.
