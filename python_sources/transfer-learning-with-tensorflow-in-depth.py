#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. Transfer learning involves the approach in which knowledge learned in one or more source tasks is transferred and used to improve the learning of a related target task.
# 
# It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems.
# 
# Traditional learning is isolated and occurs purely based on specific tasks, datasets and training separate isolated models on them. No knowledge is retained which can be transferred from one model to another. In transfer learning, you can leverage knowledge (features, weights etc) from previously trained models for training newer models and even tackle problems like having less data for the newer task!
# ### In this notebook, we will try 2 ways to customize a pretrained model:
# >**Feature Extraction**: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for our dataset.
# 
# >**Fine-Tuning**: Unfreezing a few of the top layers of a frozen model base and jointly training both the newly-added classifier layers and the last layers of the base model. This allows us to "fine tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.
# 
# **There are different transfer learning strategies and techniques, which can be applied based on the domain, task at hand, and the availability of data.**
# 
# ![image](https://miro.medium.com/max/917/1*mEHO0-LifV7MgwXSpY9wyQ.png)
# 
# <h2> References: </h2>
# <ul> 
# <li></a> https://www.tensorflow.org/tutorials/images/classification </li>
# <li></a> https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub </li>
# <li></a> https://www.tensorflow.org/tutorials/images/transfer_learning </li>
# <li></a> A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning  (Medium Article) </li>
# </ul>

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Load Data

# In[ ]:


PATH = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',extract=True, untar=True)


# ## Train and validation data set

# In[ ]:


os.listdir(PATH)


# In[ ]:


train_dir = os.path.join(PATH)
validation_dir = os.path.join(PATH)

total_train = len(os.listdir(train_dir))
total_val = len(os.listdir(validation_dir))

print("Total train  class:", total_train)
print("Total validation class:", total_val)


# In[ ]:


batch_size = 32
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150


# ## Data Preprocessing & Augmentation

# In[ ]:


# Train data set
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))


# In[ ]:


# Validation data set
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH))


# In[ ]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:


sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])


# ## Base Model Creation
# >Create MobileNet base model from the pretrained Convnets. This is pre-trained on the ImageNet dataset, a large dataset of 1.4M images and 1000 classes of web images.
# >First, you need to pick which layer of MobileNet V2 you will use for feature extraction. The very last classification layer is not very useful. So, we will use the very last layer called Bottleneck Layer, before the flatten operation.

# In[ ]:


IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# In[ ]:


feature_batch = base_model(train_data_gen.next())
print(feature_batch.shape)


# ## Feature Extraction
# >When working with a small dataset, it is common to take advantage of features learned by a model trained on a larger dataset in the same domain. This is done by instantiating the pre-trained model and adding a fully-connected classifier on top. The pre-trained model is "frozen" and only the weights of the classifier get updated during training. In this case, the convolutional base extracted all the features associated with each image and you just trained a classifier that determines the image class given that set of extracted features.
# 
# NOTE: Let us freeze the convolutional base and freeze all the layers.

# In[ ]:


base_model.trainable = False
base_model.summary()


# ### Classifier Layer

# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# In[ ]:


prediction_layer = tf.keras.layers.Dense(train_data_gen.num_classes, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# In[ ]:


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


# In[ ]:


result_batch = model.predict(sample_training_images)
result_batch.shape


# ### Compile Model

# In[ ]:


base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# ### Train Model

# In[ ]:


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // train_data_gen.batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // val_data_gen.batch_size
)


# ### Learning Plots

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.xlabel('epoch')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.show()


# ## Fine Tuning
# 
# >To increase performance even further is to fine-tune the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic features maps to features associated specifically to our dataset. In this case, you tuned your weights such that your model learned high-level features specific to the dataset. This technique is usually recommended when the training dataset is large and very similar to the original dataset that the pre-trained model was trained on.
# 
# > You should try to fine-tune a small number of top layers rather than the whole MobileNet model. In most convolutional networks, the higher up a layer is, the more specialized it is. The first few layers learn very simple and generic features which generalize to almost all types of images. As you go higher up, the features are increasingly more specific to the dataset on which the model was trained. The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.
# 
# NOTE: Perform fine tuning after you have trained the top level classifier with the pretrained model.

# ### Fine Tune from 100 layers

# In[ ]:


base_model.trainable = True
for layer in base_model.layers[:100]:
  layer.trainable =  False


# ### Compile Model

# In[ ]:


model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# ### Train Model

# In[ ]:


fine_tune_epochs = 10
total_epochs =  epochs + fine_tune_epochs

history_fine = model.fit_generator(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // train_data_gen.batch_size,
    epochs=total_epochs,
    initial_epoch = epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // val_data_gen.batch_size
)


# In[ ]:


acc = acc + history_fine.history['accuracy']
val_acc = val_acc + history_fine.history['val_accuracy']
loss = loss + history_fine.history['loss']
val_loss = val_loss + history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([epochs-1,epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([epochs-1,epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.show()

