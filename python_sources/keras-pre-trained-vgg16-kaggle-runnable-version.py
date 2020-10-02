#!/usr/bin/env python
# coding: utf-8

# Keras pre-trained VGG16 [KAGGLE RUNNABLE VERSION]
# ---------
# 
# Summary
# ---------
# 
# Some time ago I come across this nice kernel [use Keras pre-trained VGG16 acc 98%](https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98)
# The author ran this code in his computer but, it was not possible at that moment to execute it in Kaggle because some limitations:
# - There wasn't support for GPU's
# - It wasn't possible to use a pretrained model as it wasn't possible to download the data files
# 
# Nowadays these limitations don't exist and I wanted to try so, I have done some little adaptations to the mentioned kernel.
# The author asked that it took about 20s for him to process each epoc with a GTX1080ti , Kaggle uses at this moment NVIDIA Tesla K80 and it takes about 52 secods per epoch (not too bad)
# 
# References:
# - [use Keras pre-trained VGG16 acc 98%](https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98)
# - [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# - [Pretrained Keras Models: Symlinked; Not Copied ](https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied)
# - [Dogs vs. Cats Redux Playground Competition, 3rd Place Interview](http://blog.kaggle.com/2017/04/20/dogs-vs-cats-redux-playground-competition-3rd-place-interview-marco-lugo/)
# 
# 
# 

# ## resize train data and test data ##

# In[3]:


import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os

master = pd.read_csv("../input/invasive-species-monitoring/train_labels.csv")
master.head()


# In[6]:


img_path = "../input/invasive-species-monitoring/train/"

y = []
file_paths = []
for i in range(len(master)):
    file_paths.append( img_path + str(master.iloc[i][0]) +'.jpg' )
    y.append(master.iloc[i][1])
y = np.array(y)


# In[8]:


#image reseize & centering & crop 

def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized


x = []
for i, file_path in enumerate(file_paths):
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px 
    img = img[16:240, 16:240]
    x.append(img)

x = np.array(x)


# In[10]:


sample_submission = pd.read_csv("../input/invasive-species-monitoring/sample_submission.csv")
img_path = "../input/invasive-species-monitoring/test/"

test_names = []
file_paths = []

for i in range(len(sample_submission)):
    test_names.append(sample_submission.iloc[i][0])
    file_paths.append( img_path + str(int(sample_submission.iloc[i][0])) +'.jpg' )
    
test_names = np.array(test_names)


# In[13]:


test_images = []
for file_path in file_paths:
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px 
    img = img[16:240, 16:240]
    test_images.append(img)
    
    path, ext = os.path.splitext( os.path.basename(file_paths[0]) )

test_images = np.array(test_images)


# save numpy array.
# 
# Usually I separate code, data format and CNN.

# In[ ]:


#np.savez('224.npz', x=x, y=y, test_images=test_images, test_names=test_names)


# ## split train data and validation data  ##

# In[17]:


data_num = len(y)
random_index = np.random.permutation(data_num)

x_shuffle = []
y_shuffle = []
for i in range(data_num):
    x_shuffle.append(x[random_index[i]])
    y_shuffle.append(y[random_index[i]])
    
x = np.array(x_shuffle) 
y = np.array(y_shuffle)


# In[18]:


val_split_num = int(round(0.2*len(y)))
x_train = x[val_split_num:]
y_train = y[val_split_num:]
x_test = x[:val_split_num]
y_test = y[:val_split_num]

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)


# In[19]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# use Keras pre-trained VGG16
# ---------------------------
# 
# but kaggle karnel is not run

# In[ ]:


from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense

img_rows, img_cols, img_channel = 224, 224, 3

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))


# In[22]:


add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()


# In[23]:


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
train_datagen.fit(x_train)


history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    #,callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
)


# Added to plot history evolution:

# In[29]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:


print("Training loss: {:.2f} / Validation loss: {:.2f}".      format(history.history['loss'][-1], history.history['val_loss'][-1]))
print("Training accuracy: {:.2f}% / Validation accuracy: {:.2f}%".      format(100*history.history['acc'][-1], 100*history.history['val_acc'][-1]))


# ## predict test data ##

# In[24]:


test_images = test_images.astype('float32')
test_images /= 255


# In[25]:


predictions = model.predict(test_images)


# In[27]:


sample_submission = pd.read_csv("../input/invasive-species-monitoring/sample_submission.csv")

for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]

sample_submission.to_csv("submit.csv", index=False)

