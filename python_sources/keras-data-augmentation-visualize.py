#!/usr/bin/env python
# coding: utf-8

# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/5408/media/bigleaves.jpg" width="600"></img>

# ### This kernel is base on [Alexander Teplyuk](https://www.kaggle.com/ateplyuk/inat2019-starter-keras/output) here I applied Data Augmentation technic from [Udacity](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb#scrollTo=UOoVpxFwVrWy) as following:
# * random 45 degree rotation
# * random zoom of up to 50%
# * random horizontal flip
# * width shift of 0.15
# * height shfit of 0.15

# > * Model: vgg16  
#  Apply transfer learning skill from pretrained model using Keras.
# > Using vgg16 with a flatten layer followed by 2 fully connected layer with 1024 units and 1010 units. The output class probabilities based on 1010 classes which is done by the softmax activation function. Add a layer use a relu activation function. Add Dropout layers with a probability of 50%, where appropriate.

# In[ ]:


import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications import ResNet50
from keras import optimizers


# ### Train data

# In[ ]:


ann_file = '../input/train2019.json'
with open(ann_file) as data_file:
        train_anns = json.load(data_file)


# In[ ]:


train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
df_train_file_cat['category_id']=df_train_file_cat['category_id'].astype(str)
df_train_file_cat.head()


# In[ ]:


len(df_train_file_cat['category_id'].unique())


# In[ ]:


# Example of images for category_id = 400
img_names = df_train_file_cat[df_train_file_cat['category_id']=='400']['file_name'][:30]

plt.figure(figsize=[15,15])
i = 1
for img_name in img_names:
    img = cv2.imread("../input/train_val2019/%s" % img_name)[...,[2, 1, 0]]
    plt.subplot(6, 5, i)
    plt.imshow(img)
    i += 1
plt.show()


# ### Validation data

# In[ ]:


valid_ann_file = '../input/val2019.json'
with open(valid_ann_file) as data_file:
        valid_anns = json.load(data_file)


# In[ ]:


valid_anns_df = pd.DataFrame(valid_anns['annotations'])[['image_id','category_id']]
valid_anns_df.head()


# In[ ]:


valid_img_df = pd.DataFrame(valid_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
valid_img_df.head()


# In[ ]:


df_valid_file_cat = pd.merge(valid_img_df, valid_anns_df, on='image_id')
df_valid_file_cat['category_id']=df_valid_file_cat['category_id'].astype(str)
df_valid_file_cat.head()


# In[ ]:


nb_classes = 1010
batch_size = 128
img_size = 128
nb_epochs = 10


# ### In the cell below, use ImageDataGenerator to create a transformation that rescales the images by 255 here I applied Data Augmentation as following:
# * random 45 degree rotation
# * random zoom of up to 50%
# * random horizontal flip
# * width shift of 0.15
# * height shfit of 0.15

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=45, \n                    width_shift_range=.15, \n                    height_shift_range=.15, \n                    horizontal_flip=True, \n                    zoom_range=0.5)\n\ntrain_generator=train_datagen.flow_from_dataframe(\n    dataframe=df_train_file_cat,\n    directory="../input/train_val2019",\n    x_col="file_name",\n    y_col="category_id",\n    batch_size=batch_size,\n    shuffle=True,\n    class_mode="sparse",    \n    target_size=(img_size,img_size))')


# In[ ]:


# udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb#scrollTo=jqb9OGoVKIOi
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
    
augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_datagen = ImageDataGenerator(rescale=1./255)\n\nvalid_generator=test_datagen.flow_from_dataframe(\n    dataframe=df_valid_file_cat,\n    directory="../input/train_val2019",\n    x_col="file_name",\n    y_col="category_id",\n    batch_size=batch_size,\n    class_mode="sparse",    \n    target_size=(img_size,img_size))')


# In[ ]:


vgg16_net = VGG16(weights='imagenet', 
                  include_top=False, 
                  input_shape=(img_size, img_size, 3))
vgg16_net.trainable = False


# In[ ]:


resnet = ResNet50(include_top=False, weights='imagenet',
               input_shape=(img_size,img_size,3))
resnet.trainable = False


# ### Model

# In[ ]:


model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit_generator(generator=train_generator, \n                              \n                    steps_per_epoch=500,\n                              \n                    validation_data=valid_generator, \n                              \n                    validation_steps=100,\n                              \n                    epochs=nb_epochs,\n                    verbose=0)')


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(nb_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ### Test data

# In[ ]:


test_ann_file = '../input/test2019.json'
with open(test_ann_file) as data_file:
        test_anns = json.load(data_file)


# In[ ]:


test_img_df = pd.DataFrame(test_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
test_img_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_datagen = ImageDataGenerator(rescale=1./255.)\ntest_generator = test_datagen.flow_from_dataframe(      \n    \n        dataframe=test_img_df,    \n    \n        directory = "../input/test2019",    \n        x_col="file_name",\n        target_size = (img_size,img_size),\n        batch_size = 1,\n        shuffle=False,\n        class_mode = None\n        )')


# ### Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_generator.reset()\npredict=model.predict_generator(test_generator, steps = len(test_generator.filenames),verbose=1)')


# In[ ]:


len(predict)


# In[ ]:


predicted_class_indices=np.argmax(predict,axis=1)


# In[ ]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[ ]:


sam_sub_df = pd.read_csv('../input/kaggle_sample_submission.csv')
sam_sub_df.head()


# In[ ]:


filenames=test_generator.filenames
results=pd.DataFrame({"file_name":filenames,
                      "predicted":predictions})
df_res = pd.merge(test_img_df, results, on='file_name')[['image_id','predicted']]    .rename(columns={'image_id':'id'})

df_res.head()


# In[ ]:


df_res.to_csv("submission.csv",index=False)


# #### Hope you like it and finds this kernel helpful :)!

# # Reference
# [Alexander Teplyuk](https://www.kaggle.com/ateplyuk/inat2019-starter-keras/output)  
# [Udacity](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb#scrollTo=08rRJ0sn3Tb1)  
# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
