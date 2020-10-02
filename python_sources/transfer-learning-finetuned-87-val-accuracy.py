#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os 


# In[ ]:


os.listdir('/kaggle/input/danceform-identification')


# In[ ]:


train_dir = '/kaggle/input/danceform-identification/train'


# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from tensorflow.keras.applications.xception import Xception,preprocess_input
from tensorflow.keras.applications import DenseNet169,DenseNet201,MobileNetV2,ResNet50,VGG16,InceptionResNetV2,NASNetLarge
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
# from tensorflow.keras.applications.resnet import ResNet50,preprocess_input


# In[ ]:


train_df = pd.read_csv('/kaggle/input/danceform-identification/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


X_set = []
IMG_SIZE = 224
for index,row in train_df.iterrows():
    print(f'\rDone: {os.path.join(train_dir,row[0])}',end='')
    img = cv2.imread(os.path.join(train_dir,row[0]))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X_set.append(img)


# In[ ]:


X = np.array(X_set)
y = train_df['target'].values


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')


# In[ ]:


y = enc.fit_transform(train_df[['target']]).toarray()


# In[ ]:


X = X/255.0


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.imshow(X[0])
plt.title(f'{y[0]}')


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=42)


# In[ ]:


datagen = ImageDataGenerator(horizontal_flip=True,
                             rotation_range=20,
                             zoom_range=0.2,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             shear_range=0.1,
                             fill_mode="nearest")

testgen = ImageDataGenerator()

datagen.fit(X_train)
testgen.fit(X_test)


# In[ ]:


img_size = 224
base_model = DenseNet201(include_top = False,
                         weights = 'imagenet',
                         input_shape = (img_size,img_size,3))

for layer in base_model.layers[:675]:
    layer.trainable = False

for layer in base_model.layers[675:]:
    layer.trainable = True
    
# for (i,layer) in enumerate(base_model.layers):
#     print(str(i)+" "+layer.__class__.__name__,layer.trainable)


# In[ ]:


image_size = 224
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(8, activation=tf.nn.softmax))
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


filepath= "model_densenet.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

early_stopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

# learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
#                                             patience=3, 
#                                             verbose=1, 
#                                             factor=0.2, 
#                                             min_lr=0.00001)

callbacks_list = [
        checkpoint,
        early_stopping,
#         learning_rate_reduction
    ]


# In[ ]:


hist = model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),
                                        validation_data=testgen.flow(X_test,y_test,batch_size=32),
                                        epochs=50,
                                        callbacks=callbacks_list)


# In[ ]:


model = tf.keras.models.load_model('model_densenet.h5')


# In[ ]:


y_pred = model.predict(X_test)
y_pred


# In[ ]:


labels = train_df['target'].unique().tolist()
labels.sort()
print(labels)


# In[ ]:


y_ground = np.argmax(y_test,axis=1)


# In[ ]:


y_pred = np.argmax(y_pred,axis=1)
y_pred


# In[ ]:


y_true = np.argmax(y_test,axis=1)
y_true


# In[ ]:


check = y_true==y_pred
np.unique(check, return_counts=True)


# In[ ]:


plt.figure(figsize = (15 , 9))
n = 0
for i in range(len(X_test)):
    if y_pred[i] != y_true[i]:
        n+=1
        plt.subplot(5 , 5, n)
        plt.subplots_adjust(hspace = 0.8 , wspace = 0.3)
        plt.imshow(X_test[i])
        plt.title(f'Actual: {labels[y_true[i]]}\nPredicted: {labels[y_pred[i]]}')


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_ground,y_pred,target_names = labels))

