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
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
   #     print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


train_path = "../input/facespoof/dataset_2/dataset_2/train/"
test_path = "../input/facespoof/dataset_2/dataset_2/test/"
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# **Parameters**

# In[ ]:


NUM_CLASSES = 2
CHANNELS = 3
IMAGE_RESIZE = 224
POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 3
STEPS_PER_EPOCH_TRAINING = 100
STEPS_PER_EPOCH_VALIDATION = 100
BATCH_SIZE_TRAINING =743
BATCH_SIZE_VALIDATION = 185
BATCH_SIZE_TESTING = 1


# In[ ]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


# In[ ]:


model = Sequential()
model.add(MobileNetV2(include_top = False, pooling = POOLING_AVERAGE, weights = 'imagenet'))
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
model.layers[0].trainable = False


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.python.keras import optimizers
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)
image_size = IMAGE_RESIZE


# In[ ]:


from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
image_size = IMAGE_RESIZE
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


# In[ ]:




train_generator = data_generator.flow_from_directory(
        '../input/facespoof/dataset_2/dataset_2/train',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')
print(train_generator.class_indices)

validation_generator = data_generator.flow_from_directory(
        '../input/facespoof/dataset_2/dataset_2/train',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical') 


# In[ ]:


datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
TRAIN_DIR='../input/facespoof/dataset_2/dataset_2/train'
train_generator = datagen.flow_from_directory(
    TRAIN_DIR, 
    subset='training'
)
print(train_generator.class_indices)
val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    subset='validation'
)
print(train_generator.class_indices)


# In[ ]:


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')


# In[ ]:



fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs = 50,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper]
)
model.save("../working/mn2.hdf5")
model.load_weights("../working/best.hdf5")


# In[ ]:


from IPython.display import FileLink
FileLink("../working/best.hdf5")


# In[ ]:


plt.figure(1, figsize = (20,8)) 
plt.subplot(221)  
plt.plot(fit_history.history['acc'])  
plt.plot(fit_history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
plt.show()


# In[ ]:





# In[ ]:


test_generator = data_generator.flow_from_directory(
    directory = '../input/facespoof/dataset_2/dataset_2/test',
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
)


# In[ ]:


# Reset before each call to predict
#test_generator.reset()

#pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

#predicted_class_indices = np.argmax(pred, axis = 1)


# In[ ]:


""""
kk=[]
kk1=predicted_class_indices*1
for aa in range(len(test_generator.filenames)):
    if test_generator.filenames[aa][0:4]=='fake':
        kk.append(0)
    else:
        kk.append(1)
CC=0;
CC1=0;
CC2=0
CC3=0;
for aa in range(len(test_generator.filenames)):
    if kk1[aa]==kk[aa]==1:
        CC=CC+1
    if kk1[aa]==0 and kk[aa]==0:
        CC1=CC1+1
    if kk1[aa]==1 and kk[aa]==0:
        CC2=CC2+1
    if kk1[aa]==0 and kk[aa]==1:
        CC3=CC3+1
print('accuracy is', (CC+CC1)/124000)
print(CC,CC1,CC2,CC3)
""""""


# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
#import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# In[ ]:



model1 = load_model("../working/mn2.hdf5")

img = image.load_img('../input/facespoof/dataset_2/dataset_2/test/real/1050.jpg')

from timeit import default_timer as timer
start = timer()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
preds1=model.predict(x)
print(preds)
print(preds1)
end = timer()
print(end - start, 'second')


# In[ ]:


model1 = load_model('../working/mn2.hdf5')

