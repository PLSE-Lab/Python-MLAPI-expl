#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


# # **1.) Data-Preprocessing**

# In[ ]:


#Creating generator for Training DataSet
train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        '../input/100-bird-species/train',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')

#Creating generator for Validation DataSet
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_directory(
        '../input/100-bird-species/valid',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

#Creating generator for Test DataSet
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
        '../input/100-bird-species/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')


# # **2.) Build and Train CNN (VGG16)** 

# In[ ]:


#instantiate a base model with pre-trained weigts.
base_model=keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3))


# In[ ]:


#freeze the base model
base_model.trainable = False


# In[ ]:


#Create new model on top
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(2048,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.35))
model.add(Dense(2048,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.35))
model.add(Dense(200,activation='softmax',kernel_initializer='glorot_normal'))


# In[ ]:


model.summary()


# In[ ]:


#Train the model on new data.
model.compile(optimizer=keras.optimizers.Adam(1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,epochs=40,validation_data=val_generator,workers=10,use_multiprocessing=True)


# In[ ]:


#Some visualizations
import matplotlib.pyplot as plt
#Loss
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()
#Accuracy
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()


# # **Fine-Tuning**

# In[ ]:


base_model=model.layers[0]


# In[ ]:


#Un-Freezing last 2 blocks(i.e. block4 and 5)

base_model.trainable = True

set_trainable = False
for layer in base_model.layers:
    if layer.name == 'block4_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
base_model.summary()
model.summary()


# In[ ]:


model.compile(optimizer=keras.optimizers.Adam(1e-5),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


#Lets fine-tune finally....
history=model.fit(train_generator,epochs=30,validation_data=val_generator,workers=10,use_multiprocessing=True)


# In[ ]:


#few more epochs with low l_rate
model.compile(optimizer=keras.optimizers.Adam(1e-6),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,epochs=10,validation_data=val_generator,workers=10,use_multiprocessing=True)


# In[ ]:


model.save("model_fine_tuned")


# # 3.) Evaluation on Test Set

# In[ ]:


model.evaluate(test_generator,use_multiprocessing=True,workers=10)

