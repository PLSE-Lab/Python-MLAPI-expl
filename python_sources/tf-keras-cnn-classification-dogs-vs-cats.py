#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from shutil import copyfile
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
print(os.listdir("../input/dogs-vs-cats/"))


# Importing Data from this notebook: https://www.kaggle.com/abdallahhassan/dogs-cats-resnet50-transfere-learning/notebook

# In[ ]:


with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as zip_ref:
    zip_ref.extractall("train")

with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as zip_ref:
    zip_ref.extractall("test1")


# In[ ]:


train_directory = "train/train/"
test_directory  = "test1/test1/"
# See sample image
filenames = os.listdir(train_directory)
sample = random.choice(filenames)
print(sample)
image = load_img(train_directory + sample)
plt.imshow(image)


# In[ ]:


# 8000 train samples
# 1600 validation samples
import shutil
source_dir = 'train/'
def copy_files(prefix_str, range_start, range_end, target_dir):
    image_paths = []
    for i in range(range_start, range_end):
        image_path = os.path.join(source_dir,'train', prefix_str + '.'+ str(i)+ '.jpg')
        image_paths.append(image_path)
    dest_dir = os.path.join( 'data', target_dir, prefix_str)
    os.makedirs(dest_dir)

    for image_path in image_paths:
        shutil.copy(image_path,  dest_dir)

copy_files('dog', 0, 4000, 'train')
copy_files('cat', 0, 4000, 'train')
copy_files('dog', 4000, 4800,'validation')
copy_files('cat', 4000, 4800, 'validation')


# In[ ]:


# All data, 12500 cat, 12500 dog
source_dir = 'train/'
def copy_files(prefix_str, range_start, range_end, target_dir):
    image_paths = []
    for i in range(range_start, range_end):
        image_path = os.path.join(source_dir,'train', prefix_str + '.'+ str(i)+ '.jpg')
        image_paths.append(image_path)
    dest_dir = os.path.join( 'Alldata', target_dir, prefix_str)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for image_path in image_paths:
        shutil.copy(image_path,  dest_dir)

copy_files('dog', 0, 12500, 'train')
copy_files('cat', 0, 12500, 'train')


# In[ ]:


if  os.path.exists('train'):
    #os.removedirs("train")
    shutil.rmtree("train") 


# # TensorFlow Keras
# * From Convolutional Neural Networks in TensorFlow
# * https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb#scrollTo=hwHXFhVG3786

# In[ ]:


# dimensions of our images.
img_width, img_height = 150, 150

train_dir = 'data/train'
validation_dir = 'data/validation'


# In[ ]:


print(len(os.listdir('data/train/cat')))
print(len(os.listdir('data/train/dog')))
print(len(os.listdir('data/validation/cat')))
print(len(os.listdir('data/validation/dog')))


# In[ ]:


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


# In[ ]:


"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(img_width, img_height))     
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (img_width, img_height))
                                                         
                                                         """


# In[ ]:


"""
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_steps=50,
                              verbose=2)
                              """


# In[ ]:


"""
acc      = history.history[     'acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
"""


# In[ ]:


# Updated to do image augmentation

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 1000 images = batch_size * steps
      epochs=15,
      validation_data=validation_generator,
      validation_steps=50,  # 500 images = batch_size * steps
      verbose=2)


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


test_filenames = os.listdir("test1/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
print(nb_samples)


# In[ ]:


test_gen = ImageDataGenerator(rescale=1./255)
test1_generator = test_gen.flow_from_dataframe(
    test_df, 
    "test1/test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(img_width,img_height),
    batch_size=20
)


# In[ ]:


predict = model.predict_generator(test1_generator)


# In[ ]:


test_df['label'] = np.round(predict)


# In[ ]:


test_df['label'].value_counts().plot.bar()


# In[ ]:


sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['label']
    img = load_img("test1/test1/"+filename, target_size=(img_width,img_height))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


# In[ ]:


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df.drop(['filename'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)


# In[ ]:




