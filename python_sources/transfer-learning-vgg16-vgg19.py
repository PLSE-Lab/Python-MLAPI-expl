#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning
# 
# ![](https://miro.medium.com/max/3200/0*azmg0auRA-orU2GB)
# 
# Due to the training of some models, the complexity of the model or the size of the data set, it is almost impossible to perform on standard computer processors. That's why graphic processing units are needed. As a result of trainings that last for days or weeks, these trained models can be used in various ways to solve different problems. This is exactly what is called "Transfer Learning". For example; This process is advantageous when the data set you use is not large enough. If there is a model trained with a data set consisting of 15 million different images, such as ImageNet, even if your data set has very few images, much more successful results are obtained because the learning process takes place. So how many different ways can transfer learning be done?
# 
# - By freezing the whole model (trainable parameter = 0, freeze = 1) by setting the softmax output according to the number of classes of your own problem,
# - Designing by keeping several layers of the model fixed and keeping the last layers different,
# - The entire network can be used for training in your own dataset (learnable parameter = 1, freeze = 0)
# 
# Transfer learning provides faster solutions to many problems in artificial intelligence studies. Because an open source system is available and supported. The originals of the works on GitHub can be examined and used for pre-training or transfer learning in your own problems. The critical point here is to learn the applications (framework) designed for deep learning and to understand the configurations.

# ## VGG16 & VGG19
# ![](https://miro.medium.com/max/1658/1*4F-9zrU07yhwj6gChX_q-Q.png)
# It is a simple network model and the most important difference from the previous models is the use of convolutional additions with 2 or 3. It is converted into an attribute vector with 7x7x512 = 4096 neurons in the full link (FC) layer. Softmax performance of 1000 classes is calculated at the output of two FC layers. Approximately 138 million parameters are calculated. As in other models, the height and width dimensions of the matrices from the entrance to the exit decrease while the depth value (number of channels) increases.
# ![](https://miro.medium.com/max/576/0*jVx0rKGL9_u-xPaD.png)
# Filters with different weights are calculated at each convolution layer output of the model, and as the number of layers increases, the attributes formed in the filters symbolize the 'depths' of the image.
# ![](https://miro.medium.com/max/603/0*Nf7zOyC2OaNFhKIf.png)

# # Practice
# ## VGG16

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image


# In[ ]:


train_path = "../input/fruits/fruits-360/Training/"
test_path = "../input/fruits/fruits-360/Test/"


# In[ ]:


numberOfClasses = len(glob(train_path+"/*"))


# In[ ]:


from keras.applications.vgg16 import VGG16
# vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

vgg = VGG16()


# In[ ]:


vgg.summary()


# In[ ]:


vgg_layer_list = vgg.layers
vgg_layer_list


# In[ ]:


# Removing the last layer
model = Sequential()
for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])


# In[ ]:


model.summary()


# In[ ]:


for layers in model.layers:
    layers.trainable= False
    
model.add(Dense(numberOfClasses, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])


# In[ ]:


# train & test
train_data = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224))


# In[ ]:


batch_size = 32


# In[ ]:


hist = model.fit(train_data, 
                 steps_per_epoch=1600//batch_size, 
                 epochs=25, 
                 validation_data=test_data, 
                 validation_steps=800//batch_size)


# In[ ]:


model.save_weights("model.h5")


# In[ ]:


# save history
import json, codecs
with open("graph.json","w") as f:
    json.dump(hist.history,f)


# In[ ]:


# load history
with codecs.open("graph.json", encoding="utf-8") as f:
    n = json.loads(f.read())


# In[ ]:


# evaluation
plt.plot(n['loss'], label="training_loss")
plt.plot(n['val_loss'], label="validation_loss")
plt.legend()
plt.show()


# In[ ]:


plt.plot(n['accuracy'], label="training_accuracy")
plt.plot(n['val_accuracy'], label="validation_accuracy")
plt.legend()
plt.show()


# # VGG19

# In[ ]:


from keras.applications.vgg19 import VGG19
vgg = VGG19()


# In[ ]:


vgg.summary()


# In[ ]:


vgg_layer_list = vgg.layers

# Removing the last layer
model = Sequential()
for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])

    
for layers in model.layers:
    layers.trainable= False
    
model.add(Dense(numberOfClasses, activation='softmax'))


model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])


# train & test
train_data = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224))


batch_size = 32

hist = model.fit(train_data, 
                 steps_per_epoch=1600//batch_size, 
                 epochs=25, 
                 validation_data=test_data, 
                 validation_steps=800//batch_size)


# In[ ]:


# cifar10 dataset and different vgg19 method
from keras.datasets import cifar10
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

number_of_class = 10
input_shape = x_train.shape[1:]

y_train = to_categorical(y_train, number_of_class)
y_test = to_categorical(y_test, number_of_class)

# increase dimension
def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage,48,48,3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))
    return new_array

x_train = resize_img(x_train)
x_test = resize_img(x_test)

vgg = VGG19(include_top=False, weights='imagenet', input_shape=(48,48,3))

vgg_layer_list = vgg.layers

model = Sequential()
for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])
    
for layers in model.layers:
    layers.trainable= False


model.add(Flatten())
model.add(Dense(128))
model.add(Dense(number_of_class, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# train & test
train_data = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224))


hist = model.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=1000)

