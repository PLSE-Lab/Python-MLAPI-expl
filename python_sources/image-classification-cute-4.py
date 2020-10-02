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
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


PATH = os.getcwd()
print(PATH)


# In[ ]:


PATH = '../input/image-classification-cute4/10_categories/10_categories'


# In[ ]:


print(os.listdir(PATH))


# In[ ]:


data_dir_list = os.listdir(PATH)


# In[ ]:


print(data_dir_list)


# In[ ]:


#Required variables decleration and intialization


# In[ ]:


img_rows=224
img_cols=224
num_channel=3

num_epoch = 25
batch_size = 32

img_data_list=[]
classes_names_list=[]
target_column=[]


# In[ ]:


import cv2


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


#Get the number of classes


# In[ ]:


num_classes = len(classes_names_list)
print(num_classes)


# In[ ]:


#Image Pre-Processing


# In[ ]:


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255


# In[ ]:


print(img_data.shape)


# In[ ]:


num_of_samples = img_data.shape[0]
input_shape = img_data[0].shape


# In[ ]:


print(num_of_samples)


# In[ ]:


print(input_shape)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
Labelencoder = LabelEncoder()
target_column = Labelencoder.fit_transform(target_column)


# In[ ]:


target_column[0:10]


# In[ ]:


np.unique(target_column)


# **Convert class labels to numeric using One-Hot encoding**

# In[ ]:


from keras.utils import to_categorical


# In[ ]:


target_column_hotcoded = to_categorical(target_column,num_classes)


# In[ ]:


#Shuffle the dataset


# In[ ]:


from sklearn.utils import shuffle


# In[ ]:


X,Y = shuffle(img_data,target_column_hotcoded,random_state=2)


# **Train-Test Split**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_temp,y_train,y_temp = train_test_split(X,Y,test_size=0.3,random_state=2)


# In[ ]:


y_train.shape


# In[ ]:


X_test,X_val,y_test,y_val = train_test_split(X_temp,y_temp,test_size=0.3,random_state=2)


# In[ ]:


#Define the Model


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPool2D


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


# In[ ]:


first_Mod.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


first_Mod.summary()


# In[ ]:


hist = first_Mod.fit(X_train,y_train,batch_size=batch_size,epochs=num_epoch,verbose=1,validation_data=(X_test,y_test))


# In[ ]:


score = first_Mod.evaluate(X_test,y_test,batch_size=batch_size)

print('Test Loss',score[0])
print("Test Accuracy",score[1])


# In[ ]:


test_image = X_test[0:1]
print(test_image)


# In[ ]:


print(first_Mod.predict(test_image))
print(first_Mod.predict_classes(test_image))
print(y_test[0:1])


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score


# In[ ]:


Y_pred = first_Mod.predict(X_test)
print(Y_pred)
Y_train_result = first_Mod.predict(X_train)
print(Y_train_result)


# In[ ]:


y_pred = np.argmax(Y_pred,axis=1)
print(y_pred)
y_train_result = np.argmax(Y_train_result, axis=1)
print(y_train_result)


# In[ ]:


conf_Matrix = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(conf_Matrix)


# In[ ]:


y_true=np.argmax(y_test,axis=1)
y_true_train=np.argmax(y_train,axis=1)


# In[ ]:


print(classification_report(y_pred,y_true))
print(classification_report(y_train_result,y_true_train))
print("test accuracy = ",accuracy_score(y_pred,y_true))
print("validation accuracy = ",accuracy_score(y_train_result,y_true_train))


# In[ ]:


#Image Augumentation using Image Data Generator


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


# In[ ]:


TRN_AUGMENTED = os.path.join(PATH , 'Trn_Augmented_Images')
TST_AUGMENTED = os.path.join(PATH , 'Tst_Augmented_Images')


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


print('\n')

print(hist.history.keys())


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


# In[ ]:


ftrain_generator = data_gen.flow(
       X_train,
       y_train,
batch_size=batch_size,
shuffle=True,
subset='training')


# In[ ]:


ftest_generator = data_gen.flow(
X_test,
y_test,
batch_size=batch_size,
shuffle=True,
subset='validation')


# In[ ]:


first_Mod.fit_generator(ftrain_generator,epochs=num_epoch,validation_data=ftest_generator,workers=6)


# In[ ]:


first_Mod.evaluate_generator(ftest_generator,verbose=1)


# In[ ]:


#Predict on agumented dataset


# In[ ]:


#train_fdata_predict = first_Mod.predict_generator(ftest_generator,verbose=1)


# In[ ]:


#train_fdata_predict.argmax(axis=1)


# In[ ]:


Y_pred = first_Mod.predict(X_test)
print(Y_pred)


# In[ ]:


y_pred1=np.argmax(Y_pred,axis=1)
print(y_pred1)


# In[ ]:


print(confusion_matrix(np.argmax(y_test,axis=1),y_pred1))


# In[ ]:


y_true1=np.argmax(y_test,axis=1)


# In[ ]:


print(classification_report(y_pred1,y_true1))


# In[ ]:


print("test accuracy = ",accuracy_score(y_pred1,y_true1))


# In[ ]:


#Transfer Learning


# In[ ]:


from tensorflow.keras.layers import Input, Dense


# In[ ]:


image_input = Input(shape=(img_rows,img_cols,num_channel))


# In[ ]:


from tensorflow.keras.applications.vgg16 import VGG16


# In[ ]:


vgg_mod = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')


# In[ ]:


vgg_mod.summary()


# In[ ]:


last_layer = vgg_mod.get_layer('fc2').output
out = Dense(num_classes, activation= 'softmax', name = 'output')(last_layer)


# In[ ]:


from tensorflow.keras.models import Model

custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()


# In[ ]:


for layer in custom_vgg_model.layers[:-1]:
    layer.trainable= False


# In[ ]:


custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


custom_vgg_model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch,verbose=1,validation_data=(X_test,y_test))


# In[ ]:


(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


# In[ ]:


Y_train_pred = custom_vgg_model.predict(X_test)


# In[ ]:


y_train_pred = np.argmax(Y_train_pred, axis=1)
print(y_train_pred)


# In[ ]:


print(confusion_matrix(np.argmax(y_test, axis=1), y_train_pred))


# In[ ]:


y_true2=np.argmax(y_test,axis=1)


# In[ ]:


print(classification_report(y_train_pred,y_true2))


# In[ ]:


print("test accuracy = ",accuracy_score(y_train_pred,y_true2))


# In[ ]:




