#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import cv2                 
import numpy as np         
import os                  
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
# deep learning stuff
from keras.models import Sequential , Model
from keras.layers import Dense , Activation
from keras.layers import Dropout , GlobalAveragePooling2D
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD , RMSprop , Adadelta , Adam
from keras.layers import Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler
from keras.applications.inception_v3 import InceptionV3 # for transfer learning
print(os.listdir("../input/chest-xray-pneumonia/chest_xray/chest_xray/"))


# In[ ]:


TRAIN_DIR = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/"
TEST_DIR =  "../input/chest-xray-pneumonia/chest_xray/chest_xray/test/"


# Define a get label function

# In[ ]:


def get_label(Dir):
    for nextdir in os.listdir(Dir):
        if not nextdir.startswith('.'):
            if nextdir in ['NORMAL']:
                label = 0
            elif nextdir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
    return nextdir, label


# Data Pre-processing

# In[ ]:


def preprocessing_data(Dir):
    X = []
    y = []
    
    for nextdir in os.listdir(Dir):
        nextdir, label = get_label(Dir)
        temp = Dir + nextdir
        
        for image_filename in tqdm(os.listdir(temp)):
            path = os.path.join(temp + '/' , image_filename)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = skimage.transform.resize(img, (150, 150, 3))
                img = np.asarray(img)
                X.append(img)
                y.append(label)
            
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X,y


# Retrieve Data

# In[ ]:


def get_data(Dir):
    X = []
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['NORMAL']:
                label = 0 # Normal class, image is okay
            elif nextDir in ['PNEUMONIA']:
                label = 1 #abnormal Image has pneumonia
            else:
                label = 2
                
            temp = Dir + nextDir
                
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = skimage.transform.resize(img, (150, 150, 3))
                    #img_file = scipy.misc.imresize(arr=img_file, size=(299, 299, 3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)
                    
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y


# In[ ]:


X_train, y_train = get_data(TRAIN_DIR)


# In[ ]:


X_test , y_test = get_data(TEST_DIR)


# View Loaded data

# In[ ]:




print(X_train.shape,'\n',X_test.shape)
print(y_train.shape,'\n',y_test.shape)


# Convert labels to categorical data type

# In[ ]:


y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)


# Define Image set

# In[ ]:


pneumonia_images = os.listdir(TRAIN_DIR + "PNEUMONIA")
Normal_images = os.listdir(TRAIN_DIR + "NORMAL")


# Plot the images for visualization

# In[ ]:


def plotter(i):
    imagep1 = cv2.imread(TRAIN_DIR+"PNEUMONIA/"+pneumonia_images[i])
    imagep1 = skimage.transform.resize(imagep1, (150, 150, 3) , mode = 'reflect')
    imagen1 = cv2.imread(TRAIN_DIR+"NORMAL/"+Normal_images[i])
    imagen1 = skimage.transform.resize(imagen1, (150, 150, 3))
    pair = np.concatenate((imagen1, imagep1), axis=1)
    print("(Left) - No Pneumonia Vs (Right) - Pneumonia")
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
for i in range(5,10):
    plotter(i)


# In[ ]:


lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=1, verbose=1)


# Save the best model(highest validation acc) in a .h5 file using the Model checkpoint callback.

# In[ ]:


filepath="transferlearning_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# Since we have set the dimension ordering to th(Theano), we will have to reshape our dataset to a channel first order

# In[ ]:


X_train=X_train.reshape(5216,3,150,150)
X_test=X_test.reshape(624,3,150,150)


# Import the Inception V3 model without weights to allow transfer learning. Define the input shape of the images

# In[ ]:


base_diagnosis_model = InceptionV3(weights=None, include_top=False , input_shape=(3, 150, 150)) # dont include top to allow transfer 
# learning and allow custom clasification
x = base_diagnosis_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(2, activation='sigmoid')(x)


# Load the best weights attained and stored in h5. Loading the weights from the weights='imagenet' parameter was not possible hence, download weights independently and load the file.
# Download link at [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5)
# 

# In[ ]:


base_diagnosis_model.load_weights("../input/inception-weights-no-top/inception_v3_weights_notop.h5")


# There are 3 possible approaches. Allow the whole network to train,
# Make the whole of the inception network non trainable and only train the top layer
# Combine the upper 2 and train only the last layers plus the added layers on the inception model
# Define model and compile. Display model summary

# In[ ]:


model = Model(inputs=base_diagnosis_model.input, outputs=predictions)
#   for layer in base_model.layers:
#         layer.trainable = False
        
        
for layer in model.layers[:200]:
    layer.trainable = False
for layer in model.layers[200:]:
    layer.trainable = True
    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Define epochs and batch size then train the model

# In[ ]:


batch_size = 64
epochs = 10
history = model.fit(X_train, y_train, validation_data = (X_test , y_test) ,callbacks=[lr_reduce,checkpoint] ,
          epochs=epochs)


# Load the best weights saved by the ModelCheckPoint callback

# In[ ]:


model.load_weights("transferlearning_weights.hdf5")


# Plot model metrics during training and testing for visualization

# In[ ]:



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Plot confudion matrix to determine false +ves and false -ves
# We are using np.argmax since the output will be a 2x1 vector of probabilities and the index chosen is the class label which the model thinks it is the right answer.

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)
cls_report = classification_report(y_true, pred)
# print classification report
print("\n\n")
print("-"*50)
print("Report for Transfer Learning Task: ")
print("-"*50)
print(cls_report)
print("-"*50)

