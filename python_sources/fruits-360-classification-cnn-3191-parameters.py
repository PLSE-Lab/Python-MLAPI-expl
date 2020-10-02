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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import sklearn.datasets
import sklearn.model_selection
import keras.preprocessing.image
import keras.utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from skimage import color
from sklearn.metrics import accuracy_score
import keras.callbacks
import os
import numpy as np
import cv2

#def load_data(infDir):
#    infData=sklearn.datasets.load_files(infDir,load_content=False)
#    y_inf = np.array(infData['target'])
#    y_inf_names = np.array(infData['target_names'])
#    nclasses = len(np.unique(y_inf))
#    target_size=50
#    x_inf=[]
#    for filename in infData['filenames']:
#        x_inf.append(
#                keras.preprocessing.image.img_to_array(
#                        keras.preprocessing.image.load_img(filename,target_size=(target_size, target_size))
#                )
#        )
#    return([x_inf,y_inf])
    
    

train_dir = '../input/fruits-360_dataset/fruits-360/Training'
trainData=sklearn.datasets.load_files(train_dir,load_content=False)

test_dir = '../input/fruits-360_dataset/fruits-360/Test'
testData=sklearn.datasets.load_files(test_dir,load_content=False)


y_train = np.array(trainData['target'])
y_train_names = np.array(trainData['target_names'])

y_test = np.array(testData['target'])
y_test_names = np.array(testData['target_names'])

nclasses = len(np.unique(y_train))
target_size=50

x_train=[]
for filename in trainData['filenames']:
    x_train.append(
            keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(filename,target_size=(target_size, target_size))
                    )
            )
    
    
x_test=[]
for filename in testData['filenames']:
    x_test.append(
            keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(filename,target_size=(target_size, target_size))
                    )
            )


# In[ ]:


x_train=np.array(x_train)
x_train=x_train/255
y_train=keras.utils.np_utils.to_categorical(y_train,nclasses)


x_test=np.array(x_test)
x_test=x_test/255
y_test=keras.utils.np_utils.to_categorical(y_test,nclasses)


# In[ ]:


x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x_train, y_train, test_size=0.2
)
print(y_train.shape)
print(y_val.shape)


# In[ ]:


model = keras.models.Sequential()
#model.add(keras.layers.Conv2D(filters = 3, kernel_size = 1, input_shape=x_train.shape[1:],activation='tanh'))
#model.add(keras.layers.Conv2D(filters = 1, kernel_size = 1, padding='same' ,activation='sigmoid'))

model.add(keras.layers.Conv2D(filters = 4, kernel_size = (3,3), activation='relu',input_shape=(x_train.shape[1:]), name="conv_1"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters = 8, kernel_size = (3,3), activation='relu', name="conv_2"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3,3), activation='relu', name="conv_3"))
#model.add(keras.layers.Flatten())
model.add(keras.layers.pooling.GlobalAveragePooling2D(name="avg_1"))
model.add(keras.layers.Dense(nclasses,activation = 'softmax', name='output'))
model.summary()


# In[ ]:


from IPython.display import SVG
import IPython
from keras.utils import model_to_dot

print(model.summary())

keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
IPython.display.Image('test_keras_plot_model.png')


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)


# In[ ]:


history=model.fit(x_train, y_train, batch_size=64, epochs=100,validation_data=(x_val, y_val), callbacks = [checkpointer], shuffle=True)


# In[ ]:


model.load_weights('cnn_from_scratch_fruits.hdf5')


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# **Visualization** Learned patterns

# In[ ]:


import keras.backend as K


# In[ ]:


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
    
def generate_pattern(layer_name, filter_index, size=target_size):
    
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    
    input_img_data = np.random.random((1, target_size, target_size, 3)) * 20 + 128.
    
    #input_img_data = np.zeros((1, target_size, target_size, 3)) * 20 + 128.
    
    
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)


# In[ ]:


def show_patterns(layer_name):
    fig = plt.figure(figsize=(50, 50))
    for img in range(model.get_layer(layer_name).filters):
        to_show=generate_pattern(layer_name, img)
        ax = fig.add_subplot(5, 6, img+1)
        ax = plt.imshow(to_show)
        plt.xticks([])
        plt.yticks([])
        fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    


# In[ ]:


show_patterns('conv_1')


# In[ ]:


show_patterns('conv_2')


# In[ ]:


show_patterns('conv_3')


# **Visualization** Internal rappresentation

# In[ ]:


test_image = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(train_dir+"/Apple Braeburn/0_100.jpg",target_size=(target_size, target_size)))
test_image = test_image/255

plt.imshow(test_image)


# In[ ]:


fig = plt.figure(figsize=(8, 8))
for img in range(3):
    ax = fig.add_subplot(1, 3, img+1)
    ax = plt.imshow(test_image[:, :, img],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


test_image = np.expand_dims(test_image, axis=0)


# In[ ]:


hidden_rappresenter = keras.models.Model(inputs=model.input, outputs=model.get_layer('conv_1').output)
result=hidden_rappresenter.predict(test_image)
result.shape

fig = plt.figure(figsize=(8, 8))
for img in range(4):
    ax = fig.add_subplot(1, 4, img+1)
    ax = plt.imshow(result[0, :, :, img], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


# In[ ]:


hidden_rappresenter = keras.models.Model(inputs=model.input, outputs=model.get_layer('conv_2').output)
result=hidden_rappresenter.predict(test_image)
result.shape

fig = plt.figure(figsize=(8, 8))
for img in range(8):
    ax = fig.add_subplot(2, 4, img+1)
    ax = plt.imshow(result[0, :, :, img], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


# In[ ]:


hidden_rappresenter = keras.models.Model(inputs=model.input, outputs=model.get_layer('conv_3').output)
result=hidden_rappresenter.predict(test_image)
result.shape

fig = plt.figure(figsize=(8, 8))
for img in range(16):
    ax = fig.add_subplot(4, 4, img+1)
    ax = plt.imshow(result[0, :, :, img], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)


# In[ ]:


y_test_pred = model.predict(x_test)
accuracy_score(np.argmax(y_test_pred,axis=1), np.argmax(y_test,axis=1))


# In[ ]:


#    def visualize_class_activation_map(img, target_class):
#        
#       #Get the 512 input weights to the softmax.
#       class_weights = model.layers[-1].get_weights()[0]
#        
#       final_conv_layer = model.get_layer("conv2d_258")
#        
#        
#       get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
#       [conv_outputs, predictions] = get_output([img.reshape((1,50,50,3))])
#       conv_outputs = conv_outputs[0, :, :, :]
#
#       #Create the class activation map.
#       cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
#       
#       for i, w in enumerate(class_weights[:, target_class]):
#               cam += w * conv_outputs[:, :, i]
#                
#       cam /= np.max(cam)
#       cam = cv2.resize(cam, (50, 50))
#       heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
#       heatmap[np.where(cam < 0.2)] = 0
#       img = heatmap*0.5 + img      
#       
#       return(img)

