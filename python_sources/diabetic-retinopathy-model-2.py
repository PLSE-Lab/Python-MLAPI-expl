#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
import shutil


# In[ ]:


train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train['diagnosis'].unique()


# In[ ]:


train.diagnosis.hist()


# In[ ]:


def showbyserverity():
    fig = plt.figure(figsize=(25, 16))
    for class_id in sorted(train['diagnosis'].unique()): 
        for i, (idx, row) in enumerate(train.loc[train['diagnosis'] == class_id].sample(5, random_state=42).iterrows()):
            ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
            path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 50) ,-4 ,128)

            plt.imshow(image, cmap = 'gist_gray')
            ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )


# In[ ]:


showbyserverity()


# In[ ]:


def label_img(name):
    if name == 0 : 
        return np.array([1, 0, 0, 0, 0])
    elif name == 1 : 
        return np.array([0, 1, 0, 0, 0])
    elif name == 2 : 
        return np.array([0, 0, 1, 0, 0])
    elif name == 3 : 
        return np.array([0, 0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 0, 1 ])


# Copied Individual ids into differnet folders 

# In[ ]:


for class_id in sorted(train['diagnosis'].unique()):
    opath = f"/output/kaggle/working/class_{class_id}"
    os.makedirs(opath)
    for i, (idx, row) in enumerate(train.loc[train['diagnosis'] == class_id].iterrows()):
        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"
        shutil.copy(path,opath)
        
    


# In[ ]:


print(os.listdir("/output/kaggle/working"))


# In[ ]:


# shutil.rmtree("/output/kaggle/working")


# Verifying the count of individual number of images in each id

# In[ ]:


for i in range(5):
    opath = f"/output/kaggle/working/class_{i}"
    list = os.listdir(opath) # dir is your directory path
    print(len(list))
   


# In[ ]:


get_ipython().system('pip install Augmentor')


# In[ ]:


import Augmentor


# In[ ]:


def offline_augmentor(path, size, output_dir):
    p = Augmentor.Pipeline(path,output_dir)
    
    p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
#     p.shear(probability=0.5, max_shear_left=16, max_shear_right=16)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.zoom(probability=0.5, min_factor=0.75, max_factor=1.3)
    p.crop_random(probability=0.5, percentage_area=0.9)
#     p.random_brightness(probability=0.5, max_factor=1.2, min_factor=0.4)
#     p.random_color(probability=0.5, max_factor=0.8, min_factor=0.3)
#     p.random_contrast(probability=0.5, max_factor=0.8, min_factor=0.3)
    p.resize(probability=1.0, width=512, height=512)
    p.sample(size)


# In[ ]:


path = '/output/kaggle/working/class_1'
size = 1400
output_dir='/output/kaggle/working/class_1/output'

offline_augmentor(path, size, output_dir)


# In[ ]:


path = '/output/kaggle/working/class_2'
size = 800
output_dir='/output/kaggle/working/class_2/output'

offline_augmentor(path, size, output_dir)


# In[ ]:


path = '/output/kaggle/working/class_3'
size = 1600
output_dir='/output/kaggle/working/class_3/output'

offline_augmentor(path, size, output_dir)


# In[ ]:


path = '/output/kaggle/working/class_4'
size = 1500
output_dir='/output/kaggle/working/class_4/output'

offline_augmentor(path, size, output_dir)


# In[ ]:


print(os.listdir("/output/kaggle/working/class_4/output"))


# In[ ]:


path=f"/output/kaggle/working/class_4/output/class_4_original_e019b3e0f33d.png_68e673f0-8f10-4e21-bbd4-7b88e4aa10ed.png"
image = cv2.imread(path)
plt.imshow(image)


# In[ ]:


path=f"../input/aptos2019-blindness-detection/train_images/e019b3e0f33d.png"
image = cv2.imread(path)
plt.imshow(image)


# In[ ]:


train_image=[]
train_label=[]


# In[ ]:


for i in range(0,5):
    path = f"/output/kaggle/working/class_{i}"
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path_img = os.path.join(path,filename)
            image = cv2.imread(path_img)
            label = label_img(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 50) ,-4 ,128)
            train_image.append(np.array(image))
            train_label.append(label)
    if i!=0:
        path = f"/output/kaggle/working/class_{i}/output"
        for filename in os.listdir(path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                path_img = os.path.join(path,filename)
                image = cv2.imread(path_img)
                label = label_img(i)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256))
                image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 50) ,-4 ,128)
                train_image.append(np.array(image))
                train_label.append(label)    
    
        


# In[ ]:


train_image[255].shape


# In[ ]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(train_image, train_label, test_size = 0.25, random_state = 42)

xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.15, random_state=42)


# In[ ]:


type(xTrain)


# In[ ]:


xTrain = np.array(xTrain)
xTest = np.array(xTest)


# In[ ]:


yTrain = np.array(yTrain)
yTest = np.array(yTest)


# In[ ]:


xVal = np.array(xVal)
yVal = np.array(yVal)


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(train_image[123], cmap = 'gist_gray')


# In[ ]:


sumy =[]


# In[ ]:


for i in range(len(yTrain)):
    index = np.argmax(yTrain[i])
    if index == 0:        
        sumy.append(0)
    elif index == 1: 
        sumy.append(1)
    elif index == 2: 
        sumy.append(2)
    elif index == 3:
        sumy.append(3)
    elif index == 4:
        sumy.append(4)


# In[ ]:


from collections import Counter


# In[ ]:


Counter(sumy)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(5, activation = 'softmax'))


# In[ ]:


for layer in model.layers:
    print(layer.output_shape)


# In[ ]:


from keras.optimizers import SGD
from keras import metrics
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics = [metrics.categorical_accuracy],optimizer='adam')


# In[ ]:


from keras.callbacks import ModelCheckpoint


# In[ ]:


checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", monitor = 'val_categorical_accuracy',verbose=1, save_best_only=True)


# In[ ]:


history = model.fit(xTrain, yTrain, batch_size=32, epochs=40,callbacks=[checkpointer],validation_data=(xVal,yVal))


# In[ ]:


model.load_weights('best_weights.hdf5')


# In[ ]:


model.save('model_1.h5')


# In[ ]:


acc = history.history['categorical_accuracy']
loss = history.history['loss']


epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()

plt.show()


# In[ ]:


yPred = model.predict(xTest)


# In[ ]:


result = []


# In[ ]:


for i in range(len(yPred)):
    index = np.argmax(yPred[i])
    if index == 0:        #According to one hot encoding above, 0 is Coronal, 1 is Horizontal and 2 is Sagittal.
        result.append(0)
    elif index == 1: 
        result.append(1)
    elif index == 2: 
        result.append(2)
    elif index == 3:
        result.append(3)
    elif index == 4:
        result.append(4)


# In[ ]:


result_test=[]


# In[ ]:


for i in range(len(yTest)):
    index = np.argmax(yTest[i])
    if index == 0:        #According to one hot encoding above, 0 is Coronal, 1 is Horizontal and 2 is Sagittal.
        result_test.append(0)
    elif index == 1: 
        result_test.append(1)
    elif index == 2: 
        result_test.append(2)
    elif index == 3:
        result_test.append(3)
    elif index == 4:
        result_test.append(4)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_score,precision_recall_fscore_support


# In[ ]:


f1_score(result,result_test,average='macro')


# In[ ]:


confusion_matrix(result,result_test)


# In[ ]:


accuracy_score(result,result_test)


# In[ ]:


loss, accuracy = model.evaluate(xTest,yTest, batch_size=32)
print(loss, accuracy)


# Model with the best hyperparameters from previous book

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu',kernel_initializer='uniform',input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu',kernel_initializer='uniform'))
# model.add(Dropout(0.3))
model.add(Dense(5, activation = 'softmax',kernel_initializer='uniform'))


# In[ ]:


from keras.callbacks import ModelCheckpoint


# In[ ]:


from keras import metrics

model.compile(loss='categorical_crossentropy',metrics=[metrics.categorical_accuracy],optimizer='rmsprop')
checkpointer = ModelCheckpoint(filepath='best_weights.hdf5',monitor='val_categorical_accuracy',verbose=1,save_best_only=True)


# In[ ]:


history = model.fit(xTrain, yTrain, batch_size=32, epochs=40,callbacks=[checkpointer],validation_data=(xVal,yVal))


# In[ ]:


model.load_weights('best_weights.hdf5')
model.save('model_2.h5')


# In[ ]:


acc = history.history['categorical_accuracy']
loss = history.history['loss']


epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()

plt.show()


# In[ ]:


yPred = model.predict(xTest)
result = []
for i in range(len(yPred)):
    index = np.argmax(yPred[i])
    if index == 0:        #According to one hot encoding above, 0 is Coronal, 1 is Horizontal and 2 is Sagittal.
        result.append(0)
    elif index == 1: 
        result.append(1)
    elif index == 2: 
        result.append(2)
    elif index == 3:
        result.append(3)
    elif index == 4:
        result.append(4)
result_test=[]
for i in range(len(yTest)):
    index = np.argmax(yTest[i])
    if index == 0:        #According to one hot encoding above, 0 is Coronal, 1 is Horizontal and 2 is Sagittal.
        result_test.append(0)
    elif index == 1: 
        result_test.append(1)
    elif index == 2: 
        result_test.append(2)
    elif index == 3:
        result_test.append(3)
    elif index == 4:
        result_test.append(4)


# In[ ]:


recall_score(result,result_test,average='macro')


# In[ ]:


confusion_matrix(result,result_test)


# In[ ]:




