#!/usr/bin/env python
# coding: utf-8

# In[9]:


from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
import os 
from keras import backend as K 


# In[2]:


img_width, img_height = 224, 224
train_data_dir = '../input/flowers-recognition/flowers/flowers'


# In[3]:


if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 


# In[4]:


input_shape


# In[5]:


train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                shear_range = 0.2, 
                zoom_range = 0.2, 
            horizontal_flip = True,
            validation_split=0.2)


# In[6]:


test_datagen = ImageDataGenerator(rescale = 1. / 255)


# In[7]:


batch_size = 32
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                            target_size =(img_width, img_height), 
                    batch_size = batch_size, class_mode ='categorical',subset='training')

validation_generator=train_datagen.flow_from_directory(train_data_dir, 
                            target_size =(img_width, img_height), 
                    batch_size = batch_size, class_mode ='categorical',subset='validation')


# In[10]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
print(os.listdir("../input"))
weights_path = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path


# In[11]:


from keras.applications.vgg16 import VGG16
mdl = VGG16(weights=weights_path,include_top=False, input_shape = (img_width, img_height, 3))


# In[12]:


for layer in mdl.layers:
    layer.trainable = False


# In[13]:


x=mdl.output


# In[14]:


x = Flatten()(x)


# In[15]:


x = Dense(1024, activation="relu")(x)


# In[16]:


predictions = Dense(5, activation="softmax")(x)


# In[17]:


model_final = Model(input = mdl.input, output = predictions)


# In[18]:


model_final.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])


# In[19]:


model_final.fit_generator(train_generator, 
    steps_per_epoch = train_generator.samples // batch_size, 
    epochs = 1,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size)


# In[20]:


model_final.summary()


# In[21]:


model_final.save_weights('model_saved.h5') 


# In[ ]:




