#!/usr/bin/env python
# coding: utf-8

# ### Downloading the dataset from github. 

# In[ ]:


get_ipython().system('git clone https://bitbucket.org/jadslim/german-traffic-signs')


# In[ ]:


get_ipython().system('ls german-traffic-signs')


# ### Importing all the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
import random
import pickle
import pandas as pd
import cv2


# In[ ]:


np.random.seed(0)


# ### Loading the data as train, test & valadition.

# In[ ]:


with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
# TODO: Load test data
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)
    
X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']
    
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
 


# ### This step is just to make sure that I can come to know about any errors easily. One can skip this step if wanted

# In[ ]:


assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
assert(X_val.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."


# ### Visualising the data

# In[ ]:


data = pd.read_csv('german-traffic-signs/signnames.csv')
num_of_samples=[]
 
cols = 5
num_classes = 43
 
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
fig.tight_layout()
 
for i in range(cols):
    for j, row in data.iterrows():
      x_selected = X_train[y_train == j]
      axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
      axs[j][i].axis("off")
      if i == 2:
        axs[j][i].set_title(str(j) + " - " + row["SignName"])
        num_of_samples.append(len(x_selected))


# ### As we can see in graph below that this dataset is a little bit imbalanced.

# In[ ]:


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


# ### Using OpenCV2 for preprocessing the image. 

# In[ ]:



import cv2
 
plt.imshow(X_train[1000])
plt.axis("off")
print(X_train[1000].shape)
print(y_train[1000])
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# In[ ]:


img = grayscale(X_train[1000])
plt.imshow(img)
plt.axis("off")
print(img.shape)
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
img = equalize(img)
plt.imshow(img)
plt.axis("off")
print(img.shape)


# In[ ]:


def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


# In[ ]:


X_train = np.array(list(map(preprocess, X_train)))
X_test = np.array(list(map(preprocess, X_test)))
X_val = np.array(list(map(preprocess, X_val)))
 
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
print(X_train.shape)


# In[ ]:


X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)


# ### Applying the preprocessing function to every image in dataset

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
 
datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.)
 
datagen.fit(X_train)
batches = datagen.flow(X_train, y_train, batch_size = 15)
X_batch, y_batch = next(batches)
 
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
 
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(32, 32))
    axs[i].axis("off")
 
print(X_batch.shape)


# In[ ]:


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)


# ### Defining the CNN model. I have taken inspiration from the traditional LeNet architecture as I am curious how this old model performs on such a huge dataset.

# In[ ]:


def modified_model():
 model = Sequential()
 model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
 model.add(Conv2D(60, (5, 5), activation='relu'))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 
 model.add(Conv2D(30, (3, 3), activation='relu'))
 model.add(Conv2D(30, (3, 3), activation='relu'))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 
 model.add(Flatten())
 model.add(Dense(500, activation='relu'))
 model.add(Dropout(0.5))
 model.add(Dense(43, activation='softmax'))
 
 model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
 return model


# In[ ]:


model = modified_model()
print(model.summary())
 
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50),
                            epochs=10,
                            validation_data=(X_val, y_val), shuffle = 1)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
 


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','test'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.style.use('ggplot')


# ### Taking a random image from Internet to test my model

# In[ ]:


import requests
from PIL import Image
url = 'https://thumbs.dreamstime.com/t/road-signs-main-road-sign-blue-background-road-signs-main-road-sign-blue-background-109436823.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))


# In[ ]:


img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocess(img)
plt.imshow(img, cmap = plt.get_cmap('gray'))
print(img.shape)
img = img.reshape(1, 32, 32, 1)


# ### As seen below that the model is accuretly able to classify a random google image. This image belongs to the category 12, one can scroll up and find it in the data visualization part of this notebook

# In[ ]:


print("predicted sign: "+ str(model.predict_classes(img)))
# the prediction is correct.


# **I used the traditional LeNet model architecture for classifing road symbols. The model seems to work very well as I achieved validation accuracy of around 99% which is really amazing. Also I tried certain random images from the Internet to classify, the model was accuretly able to classify them. 
# I added only a single dropout layer which produced a very good accuracy. The best part of the model is that it has generalize the data very well. Would come back here and try to fine tune the model a little more to achieve 100% valadition accuracy.** 
# Anyone who is interested can copy and edit my notebook and let me know the changes. If you like my work then please upvote. 

# #### I am curious as to how this model can be deployed as an end node. I already am able to create a UI app for this but if any one have an idea as to how this can be deployed in heroku using flask then please let me know. 

# In[ ]:




