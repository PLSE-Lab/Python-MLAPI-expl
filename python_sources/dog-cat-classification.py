#!/usr/bin/env python
# coding: utf-8

# # cat dog classifier using cnn

# in below code it is only for one image to understand it properly.there are so many pictures in the folder so they all have different sizes so we have to convert them in one size for example here i have converted all the images in 64 * 64 
# size.
# 
# 
# in below both images have different size so 1st image has 400 * 400 something
# so i have scaled it in 64 * 64 size.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

datadir = "/kaggle/input/dog-and-cat-classifier/dataset/training"

CATEGORIES = ['Dog', 'Cat']

for i in CATEGORIES:
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break
img_size = 64
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []




# below code is used to go in the training dataset and  in ( try block ) convert them in same size but if they are corrupted file  in ( exception block )then pass 

# In[ ]:


def create_training_data():
    for i in CATEGORIES:

        path = os.path.join(datadir, i)
        class_num = CATEGORIES.index(i)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])

            except Exception as e:
                pass


create_training_data()
print(len(training_data))


# In[ ]:


import random

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


# # reshape the whole dataset in the form of (-1 , 64 , 64 , 1) 

# # **what is the use of -1 for reshaping ?**
# 
# 
# for example:-
# z = np.array([[1, 2, 3, 4],
#          [5, 6, 7, 8],
#          [9, 10, 11, 12]])
# z.shape
# 
# 
# 
# output = (3, 4)
# 
# 
# but if we use -1 for reshape:-
# z.reshape(-1)
# 
# 
# 
# 
# output = array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# # **now what is the use of (-1 , 1)**
# 
# for example :-
# 
# 
# output for above example but usinge reshape(-1 , 1)
# 
# 
# z.reshape(-1,1)
# 
# 
# output will be this=
# 
# 
# 
# array([[ 1],
# 
#  
#    [ 2],
#    
#    
#    [ 3],
#    
#    
#    [ 4],
#    
#    
#    [ 5],
#    
#    
#    [ 6],
#    
#    
#    [ 7],
#    
#    
#    [ 8],
#    
#   
#    [ 9],
#    
#    
#    [10],
#    
#    
#    [11],
#    
#    
#    [12]])
#    
#    
#    

# In[ ]:


X[0].reshape(-1, img_size, img_size, 1)


# In[ ]:


X = np.array(X).reshape(-1, img_size, img_size, 1)


# In[ ]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle

from keras.layers import Conv2D, MaxPooling2D


# x / 255.0 is used to normalize the data

# In[ ]:


X = X / 255.0


# In[ ]:


model= Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,y,batch_size=4,epochs=10,validation_split=0.3)

#filename = 'finalized_model.h5'
#pickle.dump(model, open(filename, 'wb'))


# In[ ]:


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

CATEGORIES=['Doggo','Cat']

image='/kaggle/input/dog-and-cat-classifier/dataset/training/Cat/9919.jpg'

def prepare(image):
    img_size=64
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)


#model = tf.keras.models.load_model("/kaggle/output/kaggle/working/finalized_model.sav")
prediction=model.predict([prepare(image)])
print(CATEGORIES[int(prediction[0][0])])
img=mpimg.imread(image)
imgplot=plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.show()


# In[ ]:


CATEGORIES=['Doggo','Cat']
image='/kaggle/input/dog-and-cat-classifier/dataset/testing/dog.4009.jpg'
def prepare(image):
    img_size=64
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)


#model = tf.keras.models.load_model("/kaggle/output/kaggle/working/finalized_model.sav")
prediction=model.predict([prepare(image)])
print(CATEGORIES[int(prediction[0][0])])
img=mpimg.imread(image)
imgplot=plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.show()


# # grid search cnn
# 
# 
# # **output is at the bottom**

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim=128, activation='relu'))
    classifier.add(Dense(output_dim=1, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)


# In[ ]:


parameters = {'batch_size' : [64, 128],
              'epochs': [50,100],
              'optimizer' : ['SGD', 'RMSprop', 'Adam']
              #'activation' : ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
             }
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           n_jobs = 1,
                           cv=5,
                           verbose=0)
grid_search = grid_search.fit(X,y,validation_split=0.3)


# In[ ]:


best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[ ]:


print(best_parameters)
print(best_accuracy)

