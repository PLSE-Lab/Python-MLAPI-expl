#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import some packages to use
import cv2
import numpy as np
import pandas as pd
from subprocess import check_output

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#To see our directory
import os
import random
import gc   #Gabage collector for cleaning deleted data from memory


# In[ ]:


train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train'
test_dir = '../input/dogs-vs-cats-redux-kernels-edition/test'

train_dogs = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]  #get dog images
train_cats = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]  #get cat images
print(train_dogs[0])
print(train_cats[0])
test_imgs = ['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images

train_imgs = train_dogs[:2000] + train_cats[:2000]  # slice the dataset and use 2000 in each class
random.shuffle(train_imgs)  # shuffle it randomly

#Clear list that are useless
del train_dogs
del train_cats
gc.collect()   #collect garbage to save memory


# In[ ]:


import matplotlib.image as mpimg
for ima in train_imgs[900:903]:
    img=mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()


# In[ ]:


#Lets declare our image dimensions
#we are using coloured images. 
nrows = 150
ncolumns = 150
channels = 3  #change to 1 if you want to use grayscale image


#A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # labels
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC)) 
        #Read the image
        #get the labels
        if 'dog' in image[-15::]:
            y.append(1)
        elif 'cat' in image[-15::]:
            y.append(0)
    
    return X, y


# In[ ]:


#get the train and label data
X, y = read_and_process_image(train_imgs)


# In[ ]:


#Lets view some of the pics
plt.figure(figsize=(20,10))
columns = 5
for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])


# In[ ]:


import seaborn as sns
del train_imgs
gc.collect()

#Convert list to numpy array
X = np.array(X)
y = np.array(y)

#Lets plot the label to be sure we just have two class
sns.countplot(y)
plt.title('Labels for Cats and Dogs')


# In[ ]:


print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)


# In[ ]:


#Lets split the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)


# In[ ]:


#clear memory
del X
del y
gc.collect()

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32


# Add pretraines model to use 

# In[ ]:


from keras.applications import InceptionResNetV2

conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))


# In[ ]:


conv_base.summary() #print out to see its 


# In[ ]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(8192, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  #Sigmoid function at the end because we have just two classes
'''
x=layers.Flatten()(conv_base.layers[-5].output)
x=layers.Dense(8192,activation='relu')(x)
x=layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=layers.Dense(512,activation='relu')(x) #dense layer 2
x=layers.Dense(256,activation='relu')(x) #dense layer 3
preds=layers.Dense(1,activation='sigmoid')(x) #final layer with softmax activation
model1=models.Model(inputs=conv_base.input,outputs=preds)
'''


# In[ ]:


#Lets see our model
model.summary()


# In[ ]:


print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))


# In[ ]:


# In[6]:#check which layer is trainable 
for i,layer in enumerate(model.layers):
    if layer.trainable==True:
        print(i,layer.name,'Ture')
    else:
        print(i,layer.name,'False')


# In[ ]:


'''
for layer in model1.layers[:775]:
    layer.trainable=False
for layer in model1.layers[775:]:
    layer.trainable=True
'''


# In[ ]:





# In[ ]:


#We'll use the RMSprop optimizer with a learning rate of 0.0001
#We'll use binary_crossentropy loss because its a binary classification
from keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])


# In[ ]:


#Lets create the augmentation configuration
#This helps prevent overfitting, since we are using a small dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale


# In[ ]:


#Create the image generators
train_generator = train_datagen.flow(X_train, y_train,batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


# In[ ]:



#The training part
#We train for 64 epochs with about 100 steps per epoch
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=20,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)


# In[ ]:


#Save the model
model.save_weights('modelTL_wieghts.h5')
model.save('modelTL_keras.h5')


# In[ ]:



#lets plot the train and val curve
import matplotlib.pyplot as plt

#get the details form the history object
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


# Let's create a function that makes our plot looks smoother and cleaner.

# In[ ]:



def smooth_plot(points, factor=0.7):
    smooth_pts = []
    for point in points:
        if smooth_pts:
            previous = smooth_pts[-1]
            smooth_pts.append(previous * factor + point * (1 - factor))
        else:
            smooth_pts.append(point)
    return smooth_pts


# In[ ]:


#Plot figure
plt.plot(epochs, smooth_plot(acc), 'b', label='Training accurarcy')
plt.plot(epochs, smooth_plot(val_acc), 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.show()


# In[ ]:


#Now lets predict on the first 10 Images of the test set
X_test, y_test = read_and_process_image(test_imgs[0:10]) #Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


i = 0
columns = 5
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
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

