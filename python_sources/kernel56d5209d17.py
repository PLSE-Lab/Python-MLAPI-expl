#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '-f')


# In[ ]:


import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import time
from PIL import Image 


# In[ ]:


img_width, img_height = 150, 150


# In[ ]:


from zipfile import ZipFile


# In[ ]:


train_data_dir = "..//input//chest-xray-pneumonia//chest_xray//train"


# In[ ]:


nb_train_samples = 40


# In[ ]:


validation_data_dir = "..//input//chest-xray-pneumonia//chest_xray//val"


# In[ ]:


nb_validation_samples = 16


# In[ ]:


batch_size = 32


# In[ ]:


epochs = 6


# In[ ]:


test_generator_samples = 40


# In[ ]:


test_batch_size = 64


# In[ ]:


K.image_data_format()


# In[ ]:


K.backend()


# In[ ]:


model = Sequential()


# In[ ]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:                                           # So, Tensorflow!
    input_shape = (img_width, img_height, 3)


# In[ ]:


model.add(Conv2D(
	             filters=24,                      
	                                               
	             kernel_size=(3, 3),               
	             strides = (1,1),                  
	                                               
	             input_shape=input_shape,          
	             use_bias=True,                     
	             padding='same',
	             name="Ist_conv_layer",
	             )
         )


# In[ ]:


model.summary()


# In[ ]:


model.add(Activation('sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.add(Conv2D(
	             filters=16,                       
	                                               
	             kernel_size=(3, 3),               
	             strides = (1,1),                  
	                                               
	             use_bias=True,                     
	             padding='same',                   
	             name="2nd_conv_layer"
	             )
         )


# In[ ]:


model.summary()


# In[ ]:


model.add(Activation('tanh'))           


# In[ ]:


model.summary()


# In[ ]:


model.add(MaxPool2D())


# In[ ]:


model.summary()


# In[ ]:


model.add(Flatten())


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(64))


# In[ ]:


model.add(Activation('relu'))    


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(32))


# In[ ]:


model.add(Activation('relu'))    


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(1))


# In[ ]:


model.add(Activation('sigmoid'))   


# In[ ]:


model.summary()


# In[ ]:


model.compile(
              loss='binary_crossentropy',  
              optimizer='rmsprop',         
              metrics=['accuracy'])     


# In[ ]:


def preprocess(img):
                    return img


# In[ ]:


tr_dtgen = ImageDataGenerator(
                              rescale=1. / 255,      
                              shear_range=0.2,       
                              zoom_range=0.2,
                              horizontal_flip=True,
                              preprocessing_function=preprocess
                              )


# In[ ]:


train_generator = tr_dtgen.flow_from_directory(
                                               train_data_dir,       
                                               target_size=(img_width, img_height),  
                                               batch_size=batch_size,  
                                               class_mode='binary'   

                                                )


# In[ ]:


val_dtgen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


validation_generator = val_dtgen.flow_from_directory(
                                                     validation_data_dir,
                                                     target_size=(img_width, img_height),   # Resize images
                                                     batch_size=batch_size,    # batch size to augment at a time
                                                     class_mode='binary'  # Return 1D array of class labels
                                                     )



# In[ ]:


start = time.time()   
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_generator:
        model.fit(x_batch, y_batch)
        batches += 1
        print ("Epoch: {0} , Batches: {1}".format(e,batches))
        if batches > 210:    
            
            break

end = time.time()
(end - start)/60


# In[ ]:


result = model.evaluate(validation_generator,
                                  verbose = 1,
                                  steps = 4        
                                  )


# In[ ]:


result  


# In[ ]:


pred = model.predict(validation_generator, steps = 2)


# In[ ]:


pred


# In[ ]:


test_data_dir =   "..//input//chest-xray-pneumonia//chest_xray//test"


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:




test_generator = test_datagen.flow_from_directory(
        test_data_dir,                         
        target_size=(img_width, img_height),   
        batch_size = test_batch_size,            
        class_mode=None)                       


# In[ ]:



im = test_generator    


# In[ ]:


images = next(im)   


# In[ ]:


images.shape    


# In[ ]:


results = model.predict(images)


# In[ ]:


results             


# In[ ]:


#Plot the images and check with
#     results
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# In[ ]:


plt.figure(figsize= (10,10))

for i in range(results.shape[0]):
    plt.subplot(5,5,i+1)
    imshow(images[i])
    
    plt.show()


# In[ ]:





# In[ ]:




