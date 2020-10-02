#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing dependecies for our use 
import pandas as pd
import numpy as np
import os 
import warnings
warnings.filterwarnings('ignore')
import cv2 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[ ]:


train=pd.read_csv("../input/train.csv")# loading the training csv file in dataframe 


# In[ ]:


train.head(5)#checking Few row of Train 


# In[ ]:


train.diagnosis.value_counts(sort=True).plot(kind='bar')
plt.show()
#this plot show the number of types of eyes problems 


# In[ ]:


train.describe() # this describe our train csv 


# In[ ]:


#checking is duplicate in train
train.duplicated().sum()


# In[ ]:


#this checking wheather Training is contain nan or not 
train.isna().sum()


# In[ ]:


image_path=os.listdir('../input/train_images')#here we extracting all name of train image 


# In[ ]:


path='../input/train_images/7b9d519cbd66.png'# here is path for single image file
print(path)# print the path for image 


# In[ ]:


# this reading the image file and reducing the image file size also
image=cv2.imread(path,1)# Reading the image using cv2 
image=cv2.resize(image,(80,80))# here we reducing the size of image for fast compution of model
plt.imshow(image)#showing the image using plt 
plt.show()


# In[ ]:


train['id_code'] = train['id_code'].apply(lambda x: "{}{}".format(x,'.png' ))#  here we append the .png format for image id 


# In[ ]:


#this loop for checking each file from  image train and train labels
y=0
for i in range(len(train.id_code)):
    for j in range(len(train.id_code)):
        if(train['id_code'][j]==image_path[i]):
            y=y+1
        
        
print("match file is ",y)  
print("train size ",len(train.id_code))
print("total image file",len(image_path))


# In[ ]:


image_path[0]==train.id_code[0]


# In[ ]:


image_path[0]
train.id_code[0]


# In[ ]:


dieases_type=[]
image_name=[]   

for i in range (len(train.id_code)):
     for j in range (len(train.id_code)):
            if(train['id_code'][j]==image_path[i]):
                dieases_type.append(train['diagnosis'][j])
                image_name.append(train['id_code'][j])
    
   
        
            
print(dieases_type[5],image_name[5])         


# In[ ]:


df2 = pd.DataFrame(list(zip(image_name, dieases_type)), columns =['image_name', 'dieases_type']) 


# In[ ]:


df2.head(4)


# In[ ]:


training_data=[]
for i in range (len(df2['image_name'])):
    path='../input/train_images/'+str(df2['image_name'][i])
    image=cv2.imread(path,cv2.IMREAD_COLOR)
    image=cv2.resize(image,(80,80))
    image= image/255
    training_data.append(image)
X = np.array(training_data)


# In[ ]:


a=10
plt.figure(figsize=(7,5))
for i in range (a):
    plt.subplot(5/a+1,a,i+1)
    plt.imshow(training_data[i])


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le=LabelEncoder()
dfle=df2['dieases_type']
dfle.category=le.fit_transform(dfle)


# In[ ]:


import keras
Y= keras.utils.to_categorical(dfle.category,5)


# In[ ]:


print("Shape of X ",X.shape)
print("shape of Y",Y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_val,y_train,y_val =train_test_split(X,Y,test_size=0.2,random_state=2)


# In[ ]:


print("Shape of X  train",X_train.shape)
print("Shape of X validate ",X_val.shape)
print("Shape of Y train",y_train.shape)
print("Shape of Y validate",y_val.shape)


# In[ ]:


batch_size=32
epochs = 200
ntrain=len(X_train)
nvalidate=len(X_val)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 


# In[ ]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(80,80,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='sigmoid'))


# In[ ]:


model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


history=model.fit(X_train,y_train, epochs=300, batch_size=32,validation_data=(X_val,y_val))


# In[ ]:


plt.figure(figsize=(10,7))
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# In[ ]:


plt.figure(figsize=(10,7))
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();


# In[ ]:


model.save("eyes.model")


# In[ ]:





# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


datagen =ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
horizontal_flip=True, fill_mode="reflect")


# In[ ]:


datagen.fit(X_train)


# In[ ]:





# In[ ]:


history=model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train) / 32, epochs=100)


# In[ ]:


plt.figure(figsize=(10,7))
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# In[ ]:


plt.figure(figsize=(10,7))
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
#plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();


# In[ ]:


from keras.models import load_model
import cv2
import numpy as np


# In[ ]:


model = load_model('eyes.model')
model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


test_path="../input/test_images/"
from tqdm.autonotebook import tqdm


# In[ ]:


test_image_name=os.listdir(test_path)


# In[ ]:


test_resut=[]
a=len(test_image_name)
for i in range(a):
    p=test_path+test_image_name[i]
    img = cv2.imread(p)
    img = cv2.resize(img,(80,80))
    img = np.reshape(img,[1,80,80,3])
    classes = model.predict_classes(img)
    test_resut.append(classes)
    


# In[ ]:


test_resut


# In[ ]:


submission = pd.DataFrame({'image_id':test_image_name,'Resut':test_resut})


# In[ ]:


submission.head(4)


# In[ ]:


filename = 'submission .csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




