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
import sklearn.neighbors

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

 
#1st convolution layer
model.add(keras.layers.Conv2D(32, (3, 3) #16 is number of filters and (3, 3) is the size of the filter.
, padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
#model.add(keras.layers.BatchNormalization())
 
#2nd convolution layer
model.add(keras.layers.Conv2D(16,(3, 3), padding='same', activation='relu')) # apply 2 filters sized of (3x3)
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
#model.add(keras.layers.BatchNormalization())

#3nd convolution layer
model.add(keras.layers.Conv2D(8,(3, 3), padding='same', activation='relu')) # apply 2 filters sized of (3x3)
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same',name='encoder'))
#model.add(keras.layers.BatchNormalization())

#here compressed version
 
#4rd convolution layer
model.add(keras.layers.Conv2D(8,(3, 3), padding='same', activation='relu')) # apply 2 filters sized of (3x3)
model.add(keras.layers.UpSampling2D((2, 2)))    
#model.add(keras.layers.BatchNormalization())

#5rd convolution layer
model.add(keras.layers.Conv2D(16,(3, 3), padding='same', activation='relu')) # apply 2 filters sized of (3x3)
model.add(keras.layers.UpSampling2D((2, 2)))
#model.add(keras.layers.BatchNormalization()) 
    
#6rd convolution layer
model.add(keras.layers.Conv2D(32,(3, 3), padding='same', activation='relu'))
model.add(keras.layers.UpSampling2D((2, 2)))
#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(3,(7, 7)))
model.add(keras.layers.Activation('sigmoid'))

model.summary()


# In[ ]:


from IPython.display import SVG
import IPython
from keras.utils import model_to_dot

print(model.summary())

keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
IPython.display.Image('test_keras_plot_model.png')


# In[ ]:


model.compile(optimizer='adadelta', loss='binary_crossentropy')
checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)


# In[ ]:


history=model.fit(x_train,
          x_train,
          batch_size=256,
          epochs=50,
          validation_data=(
               x_val,
               x_val), callbacks = [checkpointer], shuffle=True)


# In[ ]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


encoder = keras.models.Model(inputs=model.input, outputs=model.get_layer('encoder').output)
encoder.summary()


# In[ ]:


num_images = 10
random_test_images = np.random.randint(x_val.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_val)
decoded_imgs = model.predict(x_val)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_val[image_idx].reshape(50,50,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ## plot encoded image
    #ax = plt.subplot(3, num_images, num_images + i + 1)
    #plt.imshow(encoded_imgs[image_idx].reshape(28, 28))
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(50,50,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:





# In[ ]:


x_encoded_train=encoder.predict(x_train)
x_encoded_val=encoder.predict(x_val)
print(x_encoded_val.shape)


# In[ ]:


x_train_flatted=x_encoded_train.reshape(x_encoded_train.shape[0],np.prod(x_encoded_train.shape[1:]))
x_val_flatted=x_encoded_val.reshape(x_encoded_val.shape[0],np.prod(x_encoded_val.shape[1:]))


# In[ ]:


import sklearn.neighbors
knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
knn_model.fit(x_train_flatted, y_train)


# In[ ]:


x_encoded_test=encoder.predict(x_test)
x_test_flatted=x_encoded_test.reshape(x_encoded_test.shape[0],np.prod(x_encoded_test.shape[1:]))


# In[ ]:


y_test_pred = knn_model.predict(x_test_flatted)
print(accuracy_score(y_test_pred, y_test))


# **Visualization** T-SNE
# 

# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, n_iter=500)
tsne_results = tsne.fit_transform(x_val_flatted)


# In[ ]:


plt.figure(figsize=(30, 30))
plt.scatter(x=tsne_results[:,0],y=tsne_results[:,1],c=np.argmax(y_val, axis=1),cmap='rainbow')
#plt.colorbar()
plt.show()

