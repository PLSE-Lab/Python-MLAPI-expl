#!/usr/bin/env python
# coding: utf-8

# We are going to use the transfer learning technique where we are going to use the InceptionResnetV2 weights and count to get an accurate of **0.95%.**
# 
# For this kernel and for reasons of memory, we will reduce the number of training images and the number of iterations and still get a good result.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import cv2
import os
import gc

from sklearn.model_selection import train_test_split
from keras.applications import InceptionResNetV2
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img


# In[ ]:


print(os.listdir("../input/"))


# In[ ]:


train_dir = '../input/train'
test_dir = '../input/test'

# train_imgs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir)]  #get full data set
train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]  #get dog images
train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]  #get cat images

test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images


# In[ ]:


size=4000
train_imgs = train_dogs[0:size] + train_cats[0:size]


# In[ ]:


random.shuffle(train_imgs)  # shuffle it randomly


# In[ ]:


img_size = 150


# In[ ]:


def read_and_process_image(list_of_images):
    """
    Returns three arrays: 
        X is an array of resized images
        y is an array of labels
        l_id an array of Ids for submission
    """
    X = [] # images
    y = [] # labels
    l_id = [] # id for submission
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (img_size,img_size), interpolation=cv2.INTER_CUBIC))  #Read the image
        basename = os.path.basename(image)
        img_num = basename.split('.')[0]
        l_id.append(img_num)
        #get the labels
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
    
    return X, y, l_id


# In[ ]:


X, y, l_id = read_and_process_image(train_imgs)


# In[ ]:


plt.figure(figsize=(20,10))
columns = 5
for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])


# In[ ]:


X = np.array(X)
y = np.array(y)


# In[ ]:


sns.countplot(y)
plt.title('Labels for Cats and Dogs')
plt.show()


# In[ ]:


print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1)


# In[ ]:


del X
del y
del train_imgs
del train_dogs
del train_cats
gc.collect()


# In[ ]:


print("Shape of X_train",X_train.shape)
print("Shape of X_val", X_val.shape)


# In[ ]:


ntrain = len(X_train)
nval = len(X_val)


# In[ ]:


conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=[150, 150, 3]) 
conv_base.trainable = False


# In[ ]:


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))   

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[ ]:


#model.summary()


# In[ ]:


batch_size = 128  

train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    #rotation_range=30,
                                    #width_shift_range=0.2,
                                    #height_shift_range=0.2,
                                    #shear_range=0.2,
                                    #zoom_range=0.2,
                                    #horizontal_flip=True,
                                    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow(X_train, y_train,  batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


# In[ ]:


epochs = 10
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=epochs,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


# In[ ]:


X_test, y_test, l_id = read_and_process_image(test_imgs[10:20]) #Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

i = 0
columns = 5
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    pred = np.float(pred)
    if pred > 0.5:
        text_labels.append('dog ({:.3f})'.format(pred))
    else:
        text_labels.append('cat ({:.3f})'.format(pred))
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()


# In[ ]:


del X_train
del X_val
del y_train
del y_val
gc.collect()


# In[ ]:


X_test, y_test, l_id = read_and_process_image(test_imgs) 
x = np.array(X_test) / 255
del X_test

predictions = model.predict(x)
pred=pd.DataFrame(predictions, columns=['label'])
lid =pd.DataFrame(l_id, columns=['id'])

submission = pd.concat([lid,pred],axis = 1)
submission = submission.sort_values(['id'])
submission.to_csv("cats_IncepRes.csv",index=False)


# In[ ]:


binary_pred=predictions
binary_pred[predictions>0.5] = 1
binary_pred[predictions<=0.5] = 0

pred=pd.DataFrame(binary_pred, columns=['label'])
lid =pd.DataFrame(l_id, columns=['id'])

submission = pd.concat([lid,pred],axis = 1)
submission = submission.sort_values(['id'])
submission.to_csv("cats_IncepRes_bp.csv",index=False)


# In[ ]:


binary_pred=predictions
binary_pred[predictions>0.80] = 1
binary_pred[predictions<=0.2] = 0

pred=pd.DataFrame(binary_pred, columns=['label'])
lid =pd.DataFrame(l_id, columns=['id'])

submission = pd.concat([lid,pred],axis = 1)
submission = submission.sort_values(['id'])
submission.to_csv("cats_IncepRes_bp2.csv",index=False)


# In[ ]:




