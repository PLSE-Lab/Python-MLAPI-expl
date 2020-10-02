#!/usr/bin/env python
# coding: utf-8

# # Aerial Cactus Identification
# **Create a classifier capable of predicting whether an images contains a cactus.**
# ### Load libraries****

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
import os

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, Flatten,  MaxPool2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from tqdm import tqdm, tqdm_notebook


# ### Load Data
# 
# This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a columnar cactus (Neobuxbaumia tetetzo). Kaggle has resized the images from the original dataset to make them uniform in size. The file name of an image corresponds to its id.
# 
# #### Files
# 
# - train/ - the training set images
# - test/ - the test set images (you must predict the labels of these)
# - train.csv - the training set labels, indicates whether the image has a cactus (has_cactus = 1)
# - sample_submission.csv - a sample submission file in the correct format
# 

# In[ ]:


train_dir = '../input/train/train/'
test_dir = '../input/test/test/'


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
train_data.head()


# In[ ]:


print('Total training images: '+str(len(train_data)))
print(train_data.has_cactus.value_counts())


# We have total 17500 training images, among which 13136 are cactus and 4364 are non-cactus images.

# ### Data Preparation
# **Map training images to labels in train.csv**

# In[ ]:


features = []
labels = []

images = train_data['id'].values

for img_id in tqdm_notebook(images):
    features.append(cv2.imread(train_dir+img_id))
    labels.append(train_data[train_data['id'] == img_id]
                  ['has_cactus'].values[0])
    
features = np.asarray(features)
features = features.astype('float32')
features /= 255
labels = np.asarray(labels)


# ### Data Augmentation

# In[ ]:


# Data augmentation
datagen = ImageDataGenerator(
            featurewise_center=False, 
            samplewise_center= False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,  #randomly rotate image in 0 to 180 degree
            zoom_range=0.1, #randomly zoom image
            width_shift_range=0.1, #randomly shift images horizontally
            height_shift_range=0.1, #randomly shift images vertically
            horizontal_flip = False, #randomly flip images
            vertical_flip = False #randomly flip images
)
datagen.fit(features)


# ### Splitting our train and validation dataset
# 
# Now,after preprocessing is done with our data we will split our dataset to training and validation.

# In[ ]:


# split dataset
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size =0.1, random_state=2)


# ### Build the CNN model
# **[(Conv2D->relu0*3 -> MaxPool2D ]*2 -> Flatten -> Dense -> Dropout -> Dense -> Out**

# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu',
                 input_shape=(32,32,3), padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',
                 padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu',
                 padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=3, activation='relu',
                 padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu',
                 padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu',
                 padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# In[ ]:


# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


#set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 25
batch_size = 86


# In[ ]:


clf = model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size), 
                          epochs=epochs, validation_data=(X_val, y_val), verbose=2,
                          steps_per_epoch=X_train.shape[0] // batch_size, 
                          callbacks=[learning_rate_reduction])


# In[ ]:


plt.plot(clf.history['loss'])
plt.plot(clf.history['val_loss'])
plt.show()


# In[ ]:


test_features = []
test_images = []

for img_id in tqdm_notebook(os.listdir(test_dir)):
    test_features.append(cv2.imread(test_dir+img_id))
    test_images.append(img_id)
    
test_features = np.asarray(test_features)
test_features = test_features.astype('float32')
test_features /= 255


# In[ ]:


model.save('cactus_model.h5')


# In[ ]:


# Running the model over the test images

test_predictions = model.predict(test_features)
submissions = pd.DataFrame(test_predictions, columns=['has_cactus'])
submissions['has_cactus'] = submissions['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)
submissions['id'] = ''
cols = submissions.columns.tolist()
cols = cols[-1:] + cols[:-1]
submissions=submissions[cols]


# In[ ]:


for i, img in enumerate(test_images):
    submissions.set_value(i,'id',img)


# In[ ]:


# Saving the output file

submissions.to_csv('submission.csv',index=False)


# In[ ]:




