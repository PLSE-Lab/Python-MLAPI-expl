#!/usr/bin/env python
# coding: utf-8

# # **Simple Classification Benign and Malignant Skin Cancers** #
# Skin Cancers (both melanoma and non-melanoma) are some of the most common forms of cancer in the world. In 2018, there were over 1.3 million new diagnoses recorded world-wide. The countries with the highest rate of yearly melanoma diagnoses of skin are Australia and New Zealand with over 33 diagnoses per 100,000 people (see https://www.wcrf.org/dietandcancer/cancer-trends/skin-cancer-statistics).
# 
# A need exists for early detection of malignant skin cancers. Early diagnoses means early intervention and decreased risk to patients during treatment. As malignant skin cancer is commonly diagnosed visually by a specialist, classifying it presents a good opportunity to demonstrate how machine learning can be benificial to medical diagnosis.

# **Import Modules:**
# 
# We begin our notebook by importing the modules needed for the project. 
# The modules required for this project are: **tensorflow, os, numpy and matplotlib.**

# In[ ]:


import tensorflow.python as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# **Set-up Directories:**
# 
# now we set-up the filepaths for the training and validation data as seen below:

# In[ ]:


train_dir = os.path.join('../input', 'skin-cancer-malignant-vs-benign/train')
validation_dir = os.path.join('../input', 'skin-cancer-malignant-vs-benign/test')

train_benign_dir = os.path.join(train_dir, 'benign')  # directory with our training benign pictures
train_malignant_dir = os.path.join(train_dir, 'malignant')  # directory with our training malignant pictures
validation_benign_dir = os.path.join(validation_dir, 'benign')  # directory with our validation benign pictures
validation_malignant_dir = os.path.join(validation_dir, 'malignant')  # directory with our validation malignant pictures


# After setting up the filepaths for our datasets, we can use the following code to count the number of samples in each category.

# In[ ]:


num_benign_tr = len(os.listdir(train_benign_dir))
num_malignant_tr = len(os.listdir(train_malignant_dir))

num_benign_val = len(os.listdir(validation_benign_dir))
num_malignant_val = len(os.listdir(validation_malignant_dir))

total_train = num_benign_tr + num_benign_tr
total_val = num_malignant_val + num_malignant_val


# **Check Spread of Dataset:**
# 
# It is important to count the number of samples for the different types of data in our datasets to ensure that the data is not heavily biased towards one category.
# 
# **For example:** If we were to train this Neural Network with 950 sample of benign skin cancer and 100 samples of malignant skin cancers, it is more than likely that our model will be heavily biased towards predicting inputs as benign skin cancers. This would result in a model that works great on our training set, but very poorly in the real world, leading to many mis-diagnoses.
# 
# The output of the code below shows that we have an acceptable spread of data for training our neural network. With 1440 traing samples classified as benign images and 1197 samples classified as malignant.

# In[ ]:


print('total training benign images:', num_benign_tr)
print('total training malignant images:', num_malignant_tr)

print("--")
print("--")

print('total validation benign images:', num_benign_val)
print('total validation malignant images:', num_malignant_val)

print("--")
print("--")

print("Total training images:", total_train)
print("Total validation images:", total_val)


# Now that have set up the path to our dataset and ensured that we have a reasonable spread of categorized samples, we can define some of the parameters that we will use in out neural network.
# 
# **batch_size:** This parameter will define the amount of samples that we train at a time. Our model will be trained in batches until all (or most) of the training samples have been used in the epoch.
# 
# **epochs:** This is the number of times that we re-train the model to improve our hypothesis.
# 
# **IMG_HEIGHT:** the number of pixels in the vertical axis of our training samples.
# 
# 
# **IMG_WIDTH:** the number of pixels in the horizontal axis of our training samples.

# In[ ]:


batch_size = 256
epochs = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150


# **Image Data Generator:**
# 
# We continue by creating image generators for our training data and verification date. We set the rescale parameter to scale the intensity values of the pixels (RGB, Grayscale, etc) so that our network can deal with smaller numbers. This reduces the complexity of the calculations and makes the network faster to train.

# In[ ]:


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


# We then configure the image data generators. We set the shuffle parameter to true to shuffle our training data. This is important to ensure the that our training data is viewed randomly during training. We set the binary parameter as we assume that all samples will be images of a potential skin cancer and will be categorized as either malignant or benign. Lastly we set the color mode. In this instance we are assuming that the colour of a mole or skin spot is a valuable feature for predicting if it is malignant or benign.

# In[ ]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode='binary',
                                                        color_mode='rgb')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=validation_dir,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode='binary',
                                                        color_mode='rgb')


# **Display Random Samples:**
# 
# Here we display 5 randomly selected training samples to confirm that the data has been shuffled.

# In[ ]:


sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])


# **Define Model:**
# 
# Now that we have set-up our dataset, we can define the structure of our model. In this case we define a simple neural network. Dropout() is used as a way to regularize our model and reduce overfitting. In this case, 20% of neurons are randomly ignored during each training step.

# In[ ]:


model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])


# We continue to configure our model by selecting our optimizer and setting our metrics.

# In[ ]:


model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.summary()


# **Training our Model:**
# 
# Now that our model and dataset are configured we can train our model.

# In[ ]:


history = model.fit_generator(
train_data_gen,
steps_per_epoch=total_train // batch_size,
epochs=epochs,
validation_data=val_data_gen,
validation_steps=total_val // batch_size
)


# Save our model.

# In[ ]:


filename = '../output/kaggle/working/model.sav'
model.save(filename)


# And finally, plot the training and validation curves.

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
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


# **Discussion of Results:**
# 
# The model has been trained for 100 epochs. Reviewing the figures above, it is evident that the hyptothesis is approaching an asymptote at around 83% validation accuracy. The plot also shows that there is a low bias and high variance. In other words, our model is overfitting our training set. This information can be useful for deciding how to further improve our model. 
# 
# Knowing that our model has high variance, there are several assumptions that we can make to decide which actions are most likely to improve the accuracy of our model. To fix high variance, we need to add bias. The most common ways to do this are by regularization and adding more traing samples. Considering these two options we can look at the size of our training set.
# 
# Our training set consists of only 2637 images. this is considered very small for image classification. As we have already implemented some regularization in our model (using Dropout(0.2)), it would seem that adding more samples to our training samples to our dataset. In an ideal world, we would have access 10,000+ traning samples per class to make accurate models. In reality it is often difficult to source more data. In this case, we can investigate the benifits that augmentation will have on the accuracy of our model. 

# **Data Augmentation:**
# 
# When we have a small number of training samples and high variance in our model, it can be useful to randomly apply transformations to training samples to simulate more data. With tensorflow, this can be done with the Image Data Generator as shown below.

# In[ ]:


train_augmented_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


# In[ ]:


train_data_gen = train_augmented_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary',
                                                     color_mode='rgb')


# We now display one of the images that has been augmented multiple times. This is a good idea to visually check that the transformations look realistic.

# In[ ]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# To see the pure affects of data augmentation we keep the same structure for our neural network.

# In[ ]:


model_augmented = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])


# In[ ]:


model_augmented.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_augmented.summary()


# Now we train the neural network.

# In[ ]:


history = model_augmented.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# Save the model.

# In[ ]:


filename = '../output/kaggle/working/model_augmented.sav'
model_augmented.save(filename)


# And we again visually display our data.

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
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


# As predicted, augmentation of our training set resulted in an improvement in the accuracy of our model. Our Validation accuracy has improved from almost 82.81% with the standard data, to 88.30% with our augmentented data. This is a considerable improvement.
# 
# Further viewing of the figures show that the validation accuracy is still trending upwards after 100 epochs. furthermore, the validation loss is trending downwards. This implies that training our model for more epochs may result in improved accuracy of out model. 
# 
# We will now increase the number of epochs to 200 to observe any further improvements to our model.

# In[ ]:


epochs = 200


# In[ ]:


model_augmented_extended = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])


# In[ ]:


model_augmented_extended.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_augmented_extended.summary()


# In[ ]:


history = model_augmented_extended.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[ ]:


filename = 'model_augmented_extended.sav'
model_augmented.save(filename)


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
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


# **Discussion of Results:**
# 
# In this situation our validation accuracy seems to have approached a steadystate after 150 epochs. running this model for more than 150 epochs increases the likelyhood of overfitting our model to our training samples. At its peak, our model had a training accuracy of 89.65%. This occured at epoch 189. In this example we did not use any form of early stopping but doing so would save the model just before it starts to decline in performance.

# **Conclusion:**
# 
# This investigation has shown that it is possible to build a model to predict if a skin defect is malignant or benign. In its current state the model has not been developed to a standard sufficient for clinical use. To further improve this network more training samples would be most benificial, while different degrees of regularization may also prove helpful.
# 
# Neural Networks generally benifit from large, high quality datasets. However, there is no hard rule to estimate the number of training samples required to generate an accurate model. To classify a simple object, 1,000 training samples (or less) may sufficient. For a very complicated object, millions of training sample may be required. In the case of classifying malignant and benign skin cancer, I would suggest that atleast 10,000 training samples are needed per category.
