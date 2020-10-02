#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Importing the Necessary Libararies 

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPool2D
from tensorflow.keras.layers import Input, Dense
from keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16


# In[ ]:


PATH = os.getcwd()
print(PATH)
PATH = '../input/10_categories-1551435405057/10_categories'
print(os.listdir(PATH))
data_dir_list = os.listdir(PATH)
print(data_dir_list)


# #### Required variables to define the image sizes and number of epochs to run for each model 

# In[ ]:


img_rows=224
img_cols=224
num_channel=3

num_epoch = 5
batch_size = 32

img_data_list=[]
classes_names_list=[]
target_column=[]


# ##### Loading all the images from the given folder and labeling to the each image with the directory name

# In[ ]:


for dataset in data_dir_list:
    classes_names_list.append(dataset)
    print("Getting images from {} folder\n".format(dataset))
    img_list = os.listdir(PATH+'/'+ dataset)
    for img in img_list:
        input_img = cv2.imread(PATH + '/' + dataset + '/' + img)
        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
        img_data_list.append(input_img_resize)
        target_column.append(dataset)


# In[ ]:


# Checking the number of classed present 
num_classes = len(classes_names_list)
print(num_classes)


# #### Image Pre-Processing

# In[ ]:


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)


# In[ ]:


num_of_samples = img_data.shape[0]
input_shape = img_data[0].shape


# #### Label encoder to label the images with the numeric values 

# In[ ]:


Labelencoder = LabelEncoder()
target_column = Labelencoder.fit_transform(target_column)
np.unique(target_column)


# In[ ]:


# Shuffle the images and do a test train split 
target_column_hotcoded = to_categorical(target_column,num_classes)
X,Y = shuffle(img_data,target_column_hotcoded,random_state=2)
X_train,X_temp,y_train,y_temp = train_test_split(X,Y,test_size=0.3,random_state=2)
X_test,X_val,y_test,y_val = train_test_split(X_temp,y_temp,test_size=0.3,random_state=2)


# ### Building a Convolution Layer on the basic images 

# In[ ]:


first_Mod = Sequential()

first_Mod.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape))
first_Mod.add(Conv2D(64,(3,3),activation='relu'))
first_Mod.add(MaxPool2D(pool_size=(2,2)))
first_Mod.add(Dropout(0.5))

first_Mod.add(Conv2D(128,(3,3),activation='relu'))
first_Mod.add(Conv2D(128,(3,3),activation='relu'))
first_Mod.add(MaxPool2D(pool_size=(2,2)))
first_Mod.add(Dropout(0.5))

first_Mod.add(Flatten())
first_Mod.add(Dense(128,activation='relu'))
first_Mod.add(Dropout(0.5))
first_Mod.add(Dense(num_classes,activation='softmax'))


# In[ ]:


#Compile the model
first_Mod.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
first_Mod.summary()


# In[ ]:


hist = first_Mod.fit(X_train,y_train,batch_size=batch_size,epochs=num_epoch,verbose=1,validation_data=(X_test,y_test))
score = first_Mod.evaluate(X_test,y_test,batch_size=batch_size)
print('Test Loss',score[0])
print("Test Accuracy",score[1])


# In[ ]:


test_image = X_test[0:1]
plt.imshow(X_test[5])


# #### Plotting the Learning Curve 

# In[ ]:


#Summarize hist for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()

#summarize hist for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()


# #### Image Augmenta

# In[ ]:


data_gen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.5, 
    zoom_range=0.4, 
    rescale=1./255,
    vertical_flip=True, 
    validation_split=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

TRN_AUGMENTED = os.path.join(PATH , 'Trn_Augmented_Images')
TST_AUGMENTED = os.path.join(PATH , 'Tst_Augmented_Images')


# #### Image Augumentation using Image Data Generator

# In[ ]:


ftrain_generator = data_gen.flow(X_train,y_train,batch_size=batch_size,shuffle=True,subset='training')
ftest_generator = data_gen.flow(X_test,y_test,batch_size=batch_size,shuffle=True,subset='validation')


# In[ ]:


first_Mod.fit_generator(ftrain_generator,epochs=num_epoch,validation_data=ftest_generator,workers=6)


# In[ ]:


first_Mod.evaluate_generator(ftest_generator,verbose=1)


# In[ ]:


#Predict on agumented dataset


# In[ ]:


train_fdata_predict = first_Mod.predict_generator(ftest_generator,verbose=1)
train_fdata_predict.argmax(axis=1)


# In[ ]:


print("Loss: ", fd_model_evaluate[0], "Accuracy: ", fd_model_evaluate[1])


# In[ ]:


Y_pred = first_Mod.predict(X_test)
print(Y_pred[10])
plt.imshow(X_test[10])

y_pred=np.argmax(Y_pred,axis=1)
print(y_pred[10])


# In[ ]:


#Data Augmentation Using flow_from_directory


# In[ ]:


train_generator = data_gen.flow_from_directory(
        PATH,
        target_size=(img_rows, img_cols), 
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb', 
        shuffle=True,  
        #save_to_dir=TRN_AUGMENTED, 
        #save_prefix='TrainAugmented', 
        #save_format='png', 
        subset="training")


# In[ ]:


train_generator.class_indices


# In[ ]:


test_generator = data_gen.flow_from_directory(
        PATH,
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb', 
        shuffle=True, 
        seed=None, 
        #save_to_dir=TST_AUGMENTED, 
        #save_prefix='TestAugmented', 
        #save_format='png',
        subset="validation")


# In[ ]:


first_Mod.fit_generator(train_generator,epochs=num_epoch,validation_data=test_generator)
fd_model_evaluate = first_Mod.evaluate_generator(test_generator,verbose=1)
print("Loss: ", fd_model_evaluate[0], "Accuracy: ", fd_model_evaluate[1])


# In[ ]:


fd_model_predict = first_Mod.predict_generator(test_generator,verbose=1)
fd_model_predict.argmax(axis=1)


# #### Transfer Learning VGG NET

# In[ ]:


image_input = Input(shape=(img_rows,img_cols,num_channel))
vgg_mod = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
vgg_mod.summary()


# In[ ]:


last_layer = vgg_mod.get_layer('fc2').output
out = Dense(num_classes,activation='softmax',name='output')(last_layer)


# In[ ]:


cust_vgg_model = Model(image_input,out)
cust_vgg_model.summary()


# In[ ]:


for layer in cust_vgg_model.layers[:-1]:
    layer.trainable = False
cust_vgg_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])


# In[ ]:


hist_1=cust_vgg_model.fit(X_train,y_train,batch_size=batch_size,epochs=5,verbose=1,validation_data=(X_test, y_test))


# In[ ]:


Y_test_pred = cust_vgg_model.predict(X_test)
y_test_pred = np.argmax(Y_test_pred,axis=1)


# #### Plotting the Learning Curve 

# In[ ]:


#Summarize hist for accuracy
plt.plot(hist_1.history['acc'])
plt.plot(hist_1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()

#summarize hist for loss
plt.plot(hist_1.history['loss'])
plt.plot(hist_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()


# ### Manual Verification of how the classification of Images are performed on the test data 

# In[ ]:


test_generator.class_indices


# In[ ]:



for i in range(1,30):
    plt.imshow(X_test[i])
    #plt.imshow(np.fliplr(X_test[i]))
    print(y_test_pred[i])
    plt.show(block=False)
    


# In[ ]:




