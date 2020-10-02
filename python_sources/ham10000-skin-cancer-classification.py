#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as p
import PIL as pil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical,normalize
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
from keras import regularizers
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


base='/kaggle/input/skin-cancer-mnist-ham10000'


# # **Accessing Dataset**

# In[ ]:


metadata=pd.read_csv(os.path.join(base,'HAM10000_metadata.csv'))
metadata.info()


# # **Data Visualization**

# In[ ]:


#To analyse basics of Dataset we study first 5 rows of HAAM10000 Dataset
metadata.head()


# In[ ]:


#Checking for NA
metadata.isna().isna().sum()


# In[ ]:


metadata.describe(include='all')


# In[ ]:



# We need to predict the cancer type, let's check the distribution of the values
g = sns.catplot(x="dx", kind="count", palette='bright', data=metadata)
g.fig.set_size_inches(16, 5)

g.ax.set_title('Visualization of Output Classes', fontsize=20)
g.set_xlabels('Classes of Skin Cancer', fontsize=14)
g.set_ylabels('Frequency', fontsize=14)


# There is a class imbalance here with 'nv' comprising more than 65% of the overall data, this will have an impact later during the classification but we need to overcome this
# 

# In[ ]:


#Skin Cancer is confirmed via Hispathology, let's check the breakdown for each type
g = sns.catplot(x="dx", kind="count", hue="dx_type", palette='bright', data=metadata)
g.fig.set_size_inches(16, 5)

g.ax.set_title('Skin Cancer by Histopathology', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Frequency of Occurance', fontsize=14)
g._legend.set_title('Histopathology Type')


# It seems that all of 'nv' are confirmed through follow-up, this behavior is not seen for other cancer class
# 

# In[ ]:


# Skin Cancer occurence body localization
g = sns.catplot(x="dx", kind="count", hue="localization", palette='bright', data=metadata)
g.fig.set_size_inches(16, 9)

g.ax.set_title('Skin Cancer Localization', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Frequency of Occurance', fontsize=14)
g._legend.set_title('Localization')


# Skin cancer seems to have more occurence in the back and lower extrimity of people, may indicate over-exposure to the sun in water activities
# This is another great insight we get by Data Analysis

# In[ ]:


# Skin Cancer occurence by Age
g = sns.catplot(x="dx", kind="count", hue="age", palette='bright', data=metadata)
g.fig.set_size_inches(16, 9)

g.ax.set_title('Skin Cancer by Age', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Number of Data Points', fontsize=14)
g._legend.set_title('Age')


# Another big insight we get here is regarding age group having Skin Cancers
# 
# We get to know that 'nv' is predominant in people of age around 45 years old

# # Getting trainable from CSV file

# In[ ]:


df=pd.read_csv(os.path.join(base,'hmnist_28_28_RGB.csv'))
x=df.drop('label',axis=1)
y=df['label']
x=x.to_numpy()
x=x/255
y=to_categorical(y)


# In[ ]:


#Since there are 7 classes only, we can label them manually 
label={
    ' Actinic keratoses':0,
    'Basal cell carcinoma':1,
    'Benign keratosis-like lesions':2,
    'Dermatofibroma':3,
    'Melanocytic nevi':4,
    'Melanoma':6,
    'Vascular lesions':5
}


# # **Images**

# In[ ]:


x=x.reshape(-1,28,28,3)
p.figure(figsize=(50,30))
for i in range(30):
    p.subplot(10,3,i+1)
    img=x[i]
    p.imshow(img)


# # Spliting of Dataset 

# In[ ]:


trainx,trainy,testx,testy = train_test_split(x,y,test_size=0.02,random_state=42)


# # Image Augmentation to expand Train data 

# In[ ]:


data_generator=ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='nearest')


# In[ ]:


data_generator.fit(trainx)


# # CNN Model

# In[ ]:


input_shape=(28,28,3)

model=Sequential()


model.add(Conv2D(64,(2,2),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())


model.add(Conv2D(512,(2,2),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.3))

model.add(Conv2D(1024,(2,2),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())


model.add(Dropout(0.4))

model.add(Conv2D(1024,(1,1),input_shape=(28,28,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))



model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))


model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',Recall()])


# In[ ]:


#Reviewing our CNN model
model.summary()


# In[ ]:


early=EarlyStopping(monitor='accuracy',patience=4,mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1,cooldown=0, mode='auto',min_delta=0.0001, min_lr=0)


# In[ ]:


#Training our CNN
model.fit(trainx,testx,epochs=50,batch_size=90,validation_data=(trainy, testy),callbacks=[early,reduce_lr])


# In[ ]:


#Visualizing Training and Validation Accuracy
p.figure(figsize=(15,5))
loss=pd.DataFrame(model.history.history)
loss=loss[['accuracy','val_accuracy']]
loss.plot()


# In[ ]:


#Building a report 
predictions=model.predict_classes(trainy)

check=[]
for i in range(len(testy)):
  for j in range(7):
    if(testy[i][j]==1):
      check.append(j)
check=np.asarray(check)

print(classification_report(check,predictions))

