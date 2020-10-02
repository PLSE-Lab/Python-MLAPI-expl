#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Diagnosing Pneumonia from X-Rays</h1>
# 
# ![](http://www.lifeextension.com/-/media/LEF/Images/protocols/images/hero/2017_prot_pneumonia_hero.ashx?h=400&la=en&w=720&hash=C70E3107BE2504974B4B6310C654739DDE6D7D63)
# 
# 
# 
# 
# 
# Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Typically symptoms include some combination of productive or dry cough, chest pain, fever, and trouble breathing.Severity is variable.
# 
# Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications and conditions such as autoimmune diseases. Risk factors include other lung diseases such as cystic fibrosis, COPD, and asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke, or a weak immune system.Diagnosis is often based on the symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired with community, hospital, or health care associated pneumonia.
# 
# **So lets get started...**

# Starting with importing our favourite data science libraries beforehand.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

get_ipython().run_line_magic('matplotlib', 'inline')


# Now lets check out the samples that we have got in our dataset. 
# 
# 
# *Generating the list of sample images as below :-*

# In[ ]:


total_images_train_normal = os.listdir('../input/chest_xray/chest_xray/train/NORMAL/')
total_images_train_pneumonia = os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA/')


# *Now lets plot these images to get a closer look on how a normal and affected lungs look.*

# In[ ]:


sample_normal = random.sample(total_images_train_normal,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('../input/chest_xray/chest_xray/train/NORMAL/'+sample_normal[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Normal Lungs')
plt.show()


# In[ ]:


sample_pneumonia = random.sample(total_images_train_pneumonia,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('../input/chest_xray/chest_xray/train/PNEUMONIA/'+sample_pneumonia[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Pneumonia Lungs')
plt.show()


# <h2>Thats great till now !</h2>
# 
# *Now lets analyze the ratio of Normal and pneumonia affected lungs from the dataset.*

# In[ ]:


sns.set_style('whitegrid')
sns.barplot(x=['Normal','Pneumonia'],y=[len(total_images_train_normal),len(total_images_train_pneumonia)])


# Take Away :- Pneumonia affected samples are quite more than the normal ones in the dataset.

# **Thats done ! And we are ready to proceed with the creation of our model.**
# 
# *We start with defining some of the constants that we will be using during the model creation phase.*

# In[ ]:


image_height = 150
image_width = 150
batch_size = 10
no_of_epochs  = 10


# *Note :-  Number of Epochs is taken as 10 for demo purpose.You can increase this value to get more accurate results.*

# <h2>Now we start the 'BIG GAME' . Creating the architecture of the model.</h2>

# In[ ]:


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(image_height,image_width,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# *Lets analyse our model architecture now *

# In[ ]:


model.summary()


# Lets do some Data Augmentation before feeding our image in our CNN model. Performing some data augmentation increases the training data and thus helps in reducing overfitting  and gives better results.However one should also consider that this process reduces the processing time as well so one should choose the proper augmentations as per the requirements.
# 
# *We apply Rescaling,Shearing,zooming and Rotation on our **Training set** and only rescaling on **Test set**.*

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   shear_range=0.2,
                                   zoom_range=0.2
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)


# *Now getting the images from the Dataset.*

# In[ ]:


training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',
                                                 target_size=(image_width, image_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test',
                                            target_size=(image_width, image_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

# Updated part --->
val_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/val',
                                            target_size=(image_width, image_height),
                                            batch_size=1,
                                            shuffle=False,
                                            class_mode='binary')


# We also add ReduceLROnPlateau callback on our model.
# 
# Applying ReduceLROnPlateau  reduces the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

# In[ ]:


reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]


# <h2>And the TRAINING STARTS...</h2>

# In[ ]:


history = model.fit_generator(training_set,
                    steps_per_epoch=5216//batch_size,
                    epochs=no_of_epochs,
                    validation_data=test_set,
                    validation_steps=624//batch_size,
                    callbacks=callbacks
                   )


# In[ ]:


# display indices marked by the system

print(test_set.class_indices)


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(16,9))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()


# In[ ]:


predictions = model.predict_generator(val_set, steps=16, verbose=1)


# In[ ]:


predictions.shape


# <h2 align="center">Thats All for now ! Stay tuned for updates !!!</h2>
# 
# 
# 
# <h1 align="center">Thanks for your time !</h1>
