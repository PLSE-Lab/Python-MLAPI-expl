#!/usr/bin/env python
# coding: utf-8

# This kernel is a follow-up from the notebook presenting a CNN for classification of cacti in aerial imagery (https://www.kaggle.com/frlemarchand/simple-cnn-using-keras). While this previous work aimed to simply use a CNN for a fairly easy binary task, the same architecture will not be used as easily on such large images as for this current cancer detection task. Of course, some code will be similar.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import os
from shutil import copyfile, move
from tqdm import tqdm
import h5py
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16


# In[ ]:


dataset_df = pd.read_csv("../input/train_labels.csv")
dataset_df["filename"] = [item.id+".tif" for idx, item in dataset_df.iterrows()]
dataset_df["groundtruth"] = ["cancerous" if item.label==1 else "healthy" for idx, item in dataset_df.iterrows()]
dataset_df.head()


# We create the training and validation sets. The training set is composed of 80% of the dataset and the validation set contains the 20% left. It is important to have a large enough validation set as some of our training conditions (for example, early stopping) relies on the performance on the validation set.

# In[ ]:


training_sample_percentage = 0.8
training_sample_size = int(len(dataset_df)*training_sample_percentage)
validation_sample_size = len(dataset_df)-training_sample_size

training_df = dataset_df.sample(n=training_sample_size)
validation_df = dataset_df[~dataset_df.index.isin(training_df.index)]


# # Load the dataset

# While for smaller datasets, it is okay to open and load images into memory, we reach a grand total of 220025 images here. It also means that copying the files into well ordered folders becomes a bit of a long mess. Therefore, we can use the *flow_from_dataframe* method from *ImageDataGenerator* to associate the existing labels from the "train_labels.csv" file with the images provided: Simple and extremely efficient.
# Please note that we can here afford to play around with data augmentation due to the nature of the dataset. Indeed, only the 32x32-pixel area in the centre of our 96x96-pixel images can contain cancerous cells, as described in the documentation.

# In[ ]:


training_batch_size = 64
validation_batch_size = 64
target_size = (96,96)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2, 
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = training_df,
    x_col='filename',
    y_col='groundtruth',
    directory='../input/train/',
    target_size=target_size,
    batch_size=training_batch_size,
    shuffle=True,
    class_mode='binary')


validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    x_col='filename',
    y_col='groundtruth',
    directory='../input/train/',
    target_size=target_size,
    shuffle=False,
    batch_size=validation_batch_size,
    class_mode='binary')


# Even though the approach has already been decided, it can be insightful to visually assess the images constituting the dataset. Moreover, it is the perfect opportunity to check whether any data augmentation may have generated strange-looking images.

# In[ ]:


def plot_random_samples(generator):
    generator_size = len(generator)
    index=random.randint(0,generator_size-1)
    image,label = generator.__getitem__(index)

    sample_number = 10
    fig = plt.figure(figsize = (20,sample_number))
    for i in range(0,sample_number):
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(image[i])
        if label[i]==0:
            ax.set_title("Cancerous cells")
        elif label[i]==1:
            ax.set_title("Healthy cells")
    plt.tight_layout()
    plt.show()


# In[ ]:


plot_random_samples(validation_generator)


# # Create the model: Use a pretrained VGG16

# The VGG16 is placed at the beginning of our model, which pre-initialised weights based on the training on the ImageNet dataset. A layer is added at the very end of the model to learn the classification between the two classes. Also, I decided to unfreeze the last layers' weights of the imported VGG16. The type of images in cancer research we are using are very specific and not represented in ImageNet.
# More reading can be done on transfer learning strategies there: https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751
# 
# While I initially attempted to use a ResNet50 for this kernel, it turned out that I could not freeze only a part of the layers without leading to very unexpected behaviours. As this kernel was written to demonstrate this very particular transfer learning strategy, I went for a VGG16 despite that the ResNet50 could reach results around 96-97% of accuracy by leaving all the layers trainable but taking longer to train.
# 
# More reading about the different pretrained architectures can be found here: https://towardsdatascience.com/neural-network-architectures-156e5bad51ba

# In[ ]:


input_shape = (96, 96, 3)
pretrained_layers = VGG16(weights='imagenet',include_top = False, input_shape=input_shape)
pretrained_layers.summary()


# We freeze all the layers except the 8 last, before checking the "trainable" status of the all the layers in our VGG16.

# In[ ]:


for layer in pretrained_layers.layers[:-8]:
    layer.trainable = False

for layer in pretrained_layers.layers:
    print(layer, layer.trainable)


# We proceed to create the model by adding the pretrained VGG16 and then our bottleneck layers which will finish with a binary classification.

# In[ ]:


dropout_dense_layer = 0.6

model = Sequential()
model.add(pretrained_layers)
    
model.add(GlobalAveragePooling2D())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[ ]:


model.summary()


# The learning rate is set at 0.001, which could be considered as high. However, one of the callback functions checks whether the loss on the validation is going down and will automatically decrease the learning rate if the loss stagnates.

# In[ ]:


model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])


# In[ ]:


callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5),
             EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

train_step_size = train_generator.n // train_generator.batch_size
validation_step_size = validation_generator.n // validation_generator.batch_size


# In[ ]:


epochs = 20
history = model.fit_generator(train_generator,
          steps_per_epoch = train_step_size,
          validation_data= validation_generator,
          validation_steps = validation_step_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          callbacks=callbacks)


# # Plotting performance during training

# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy over epochs')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# # Load the best model and classify images from the test set

# In[ ]:


model.load_weights("best_model.h5")


# Once again, we use the ImageDataGenerator to load our images. This time, I only used the `flow_from_directory` as we do not need to associate labels to the images.

# In[ ]:


src="../input/test"

test_folder="../test_folder"
dst = test_folder+"/test"
os.mkdir(test_folder)
os.mkdir(dst)

file_list =  os.listdir(src)
with tqdm(total=len(file_list)) as pbar:
    for filename in file_list:
        pbar.update(1)
        copyfile(src+"/"+filename,dst+"/"+filename)
        
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    directory=test_folder,
    target_size=target_size,
    batch_size=1,
    shuffle=False,
    class_mode='binary'
)


# In[ ]:


pred=model.predict_generator(test_generator,verbose=1)


# In[ ]:


csv_file = open("sample_submission.csv","w")
csv_file.write("id,label\n")
for filename, prediction in zip(test_generator.filenames,pred):
    name = filename.split("/")[1].replace(".tif","")
    csv_file.write(str(name)+","+str(prediction[0])+"\n")
csv_file.close()

