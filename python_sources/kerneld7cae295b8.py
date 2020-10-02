#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard

import numpy as np
import PIL
import os
import matplotlib.pyplot as plt


# In[2]:


def set_trainable(model, flag = False):
    for layer in model.layers:
        layer.trainable = flag


# In[3]:


def path_join(dirname, filenames):
    return[os.path.join(dirname,filename) for filename in filenames]


# In[4]:


batch_size = 20
num_classes = 2
epochs = 5


# In[5]:


get_ipython().system('ls ../input/pleasedetect/pleasedetect')


# In[6]:


train_dir = '../input/pleasedetect/pleasedetect/training/'
val_dir = '../input/pleasedetect/pleasedetect/validation'


# In[7]:


input_shape = (300,300)


# In[8]:


data_gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=180, zoom_range=[0.9,1.3])


# In[9]:


data_gen_val = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[10]:


generator_train = data_gen_train.flow_from_directory(directory=train_dir, target_size=input_shape, batch_size=70, shuffle=True)


# In[11]:


generator_val = data_gen_val.flow_from_directory(directory=val_dir, target_size=input_shape, batch_size=10, shuffle=True)


# In[12]:


pre_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(300,300,3))
pre_model.summary()


# In[13]:


last_pre_layers = ['block5_pool']


# In[14]:


for pre_layer in last_pre_layers:
    
    pre_layer_output = pre_model.get_layer(pre_layer)
    ref_model = keras.models.Model(inputs=pre_model.input, outputs=pre_layer_output.output)
    
    set_trainable(ref_model, flag=False)
    
    dense_values = [1024]
    
    for dense_val in dense_values:
        
        NAME = "x-lr-05-ep-200-retro-pre_layer-{}-dense-{}-time-{}".format(pre_layer, dense_val, int(time.time()))
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#         transfer_model.add(tf.layers.batch_normalization(input=ref_model))
        transfer_model = keras.models.Sequential()
        transfer_model.add(ref_model)
        transfer_model.add(keras.layers.Flatten())
        transfer_model.add(keras.layers.Dense(512, activation='relu'))
        transfer_model.add(keras.layers.Dense(128, activation='relu'))
        transfer_model.add(keras.layers.Dense(64, activation='relu'))
        transfer_model.add(keras.layers.Dense(8, activation='relu'))
        transfer_model.add(keras.layers.Dense(2, activation='softmax'))
        optimizer = keras.optimizers.Adam(lr=0.0001)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        transfer_model.compile(optimizer=optimizer, loss = loss, metrics=metrics)

        #history = transfer_model.fit_generator(generator=generator_train, epochs=epochs, steps_per_epoch=5, validation_data=generator_val, validation_steps=3)
        transfer_model.fit(generator_train, epochs=20, steps_per_epoch=5, validation_data=generator_val, validation_steps=5)
        #keras.models.save_model(transfer_model, filepath='../input/{}.model'.format(NAME))


# In[15]:


with open("model.json", "w") as file:
    file.write(transfer_model.to_json())
keras.models.save_model(transfer_model, filepath='./{}.h5'.format(NAME))


# In[16]:


import numpy as np
from keras.preprocessing import image


# In[22]:


test_image = image.load_img('../input/pleasedetect/pleasedetect/20190506_180804_003 resized 128.jpg',target_size=(300,300))
test_image = image.img_to_array(test_image)
test_image= np.expand_dims(test_image,axis = 0)
result = transfer_model.predict(test_image)
if result[0][0] >= 0.5:
    prediction = "1 USD"
elif result[0][1] >= 0.5:
    prediction = "100 Rupee"
else:
    prediction = "not predicted"
get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('../input/pleasedetect/pleasedetect/20190506_180804_003 resized 128.jpg')
imgplot = plt.imshow(img)
plt.show()
print(prediction)


# In[18]:


get_ipython().system("ls '../input/pleasedetect/pleasedetect/'")


# In[23]:


test_image = image.load_img('../input/pleasedetect/pleasedetect/IMG_20190510_091318 128.jpg',target_size=(300,300))
test_image = image.img_to_array(test_image)
test_image= np.expand_dims(test_image,axis = 0)
result = transfer_model.predict(test_image)
if result[0][0] >= 0.5:
    prediction = "1 USD"
elif result[0][1] >= 0.5:
    prediction = "100 Rupee"
else:
    prediction = "not predicted"
get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('../input/pleasedetect/pleasedetect/IMG_20190510_091318 128.jpg')
imgplot = plt.imshow(img)
plt.show()
print(prediction)


# In[ ]:




