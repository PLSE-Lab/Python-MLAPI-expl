#!/usr/bin/env python
# coding: utf-8

# # A try to the dogs vs cats dataset using a simple CNN
# This has been one of my first ML projects.
# 
# I got inspired by [sentdesk](https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/), [Adrian Rosebrock](https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/) and [Uysim Ty](https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification). Thanks!
# 
# Any feedback would be great :)

# # Import necessary packages

# In[ ]:


import numpy as np
import os
import cv2
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import random
from mlxtend.plotting import plot_confusion_matrix
from tqdm import tqdm


# # Let's have a look of where the data is

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # So we need to extract the images in the zip files

# In[ ]:


with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/train.zip') as z:
    z.extractall(".")

with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/test.zip') as z:
    z.extractall(".")

print(os.listdir('.'))


# # Now, we have two folders containing each train and test images
# # The next step is to load the training data
# ## Both features (images) and labels (dog or cat) are loaded into a python list

# In[ ]:


DATADIR = './train'
training_data = []
RESIZE = 100
X = []
y = []

def create_training_data():
    for img in os.listdir(DATADIR):
        try:
            img_array = cv2.imread(os.path.join(DATADIR,img), cv2.IMREAD_GRAYSCALE)            
            img2 = cv2.resize(img_array, (RESIZE,RESIZE))
            img2 = (img2 - img2.mean())/img2.std()
            if img[:3] == 'dog':
                class_num = 0
            else:
                class_num = 1
            X.append(img2)
            y.append(class_num)
        except Exception as e:
            pass
        
create_training_data()


# ## The python list containing the loaded data is converted into two numpy arrays, one for features and one for labels

# In[ ]:


X = np.array(X).reshape(-1, RESIZE, RESIZE, 1)
y = np.asarray(y)


# ## Now, we divide the training data into two sets, one for training and one for validation

# In[ ]:


(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.3, random_state=42)


# # Now, it is time to build and train a simple CNN model

# ## First, we create generators for augmentation of training data and for normalization of validation data

# In[ ]:


aug_train = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

generator_val = ImageDataGenerator()


# ## The ImageDataGenerator.fit method is used for feature normalization

# In[ ]:


aug_train.fit(X_train)

generator_val.fit(X_val)


# ## We build now a CNN. Let's try with a simple one consisnting in 5 Conv layer, one dense layer and one ouput layer

# In[ ]:


model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# ## Ok, let's now train the model

# In[ ]:


earlystop = EarlyStopping(patience=5)

history = model.fit(aug_train.flow(X_train, y_train, batch_size=32), validation_data=generator_val.flow(X_val, y_val, batch_size=32), epochs=100, callbacks=[earlystop])


# ## Evaluate model accuracy on the validation dataset

# In[ ]:


arr = model.predict(X_val.astype(float))
predicted_label = np.argmax(arr, axis=1)
print("Model accuracy on validation set: {:.4f}".format(accuracy_score(y_val, predicted_label)))


# ## Let's have a look at the correlation matrix

# In[ ]:


cm  = confusion_matrix(y_val, predicted_label)
plot_confusion_matrix(cm,figsize=(6,6), cmap=plt.cm.Blues, colorbar=True)
plt.xticks(range(2), ['Dogs', 'Cats'], fontsize=16)
plt.yticks(range(2), ['Dogs', 'Cats'], fontsize=16)
plt.show()


# ## Let's now generate a submission file according to the instructions

# In[ ]:


TESTDIR = './test'
LABELS = ["DOG", "CAT"]
test_data = []
RESIZE = 100
X_test = []
X_id = []

def create_test_data():
    for img in os.listdir(TESTDIR):
        try:
            img_array = cv2.imread(os.path.join(TESTDIR,img), cv2.IMREAD_GRAYSCALE)            
            img2 = cv2.resize(img_array, (RESIZE,RESIZE))
            X_test.append(img2)
            img_num = img.split('.')[0]
            X_id.append(np.array(img_num))
            
        except Exception as e:
            pass
        
create_test_data()
X_test = np.array(X_test).reshape(-1, RESIZE, RESIZE, 1)


arr_test = model.predict(X_test.astype(float))


# In[ ]:


submission = pd.DataFrame({'id':X_id,'label':arr_test[:,0]})


# In[ ]:


submission.head()


# In[ ]:


filename = 'Prediction1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# ## I'm too late for the competition, but let's see how the model works for some examples of the test data s

# In[ ]:


test_predicted_label = np.argmax(arr_test, axis=1)
fig=plt.figure(figsize=(20,20))

for counter, img in enumerate(X_test[:40]):
    ax = fig.add_subplot(10,4,counter+1)
    ax.imshow(X_test[counter,:,:,0], cmap='gray')
    plt.title(LABELS[test_predicted_label[counter]])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
plt.tight_layout()
plt.show()

