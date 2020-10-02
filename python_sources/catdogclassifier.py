#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/setting-up-kaggle-in-google-colab-ebb281b61463

# ## Setup

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install -q kaggle')


# In[ ]:


from google.colab import files
# Upload kaggle API key file (kaggle.json)
uploaded = files.upload()


# In[ ]:


get_ipython().run_line_magic('cd', '/root')
get_ipython().system('mkdir -p .kaggle')
get_ipython().system('mv /content/kaggle.json /root/.kaggle/kaggle.json')
get_ipython().run_line_magic('cd', '/content')


# In[ ]:


get_ipython().run_line_magic('cd', '/root/.kaggle/')
get_ipython().system('ls')
get_ipython().run_line_magic('cd', '/content')


# ## Download Data

# In[ ]:


get_ipython().system('kaggle competitions download -c dogs-vs-cats')


# In[ ]:


get_ipython().system('unzip test1.zip')
get_ipython().system('unzip train.zip')
get_ipython().run_line_magic('cd', '/content')
get_ipython().system('mkdir data')
get_ipython().system('mv /content/test1 /content/data/test1')
get_ipython().system('mv /content/train /content/data/train')
get_ipython().system('mv /content/sampleSubmission.csv /content/data/sampleSubmission.csv')
get_ipython().system('rm -fr /content/test1.zip')
get_ipython().system('rm -fr /content/train.zip')


# ## Import Libraries

# In[ ]:


import os 
import cv2
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ## Analyze Data

# kaggle/input/dogs-vs-cats/train/train/

# In[ ]:


get_ipython().run_line_magic('cd', 'kaggle/input/dogs-vs-cats/train/train/')


# In[ ]:


path = '/kaggle/input/dogs-vs-cats/train/train/'
file_names, width_list, height_list, ratios_list = ([] for _ in range(4))
shapes_dict = {}
count = 0
for file in os.listdir(path):
  img = cv2.imread(f'{path}/{file}', 0)
  file_names.append(file)
  width_list.append(img.shape[0])
  height_list.append(img.shape[1])
  ratios_list.append(img.shape[0]/img.shape[1])
  shapes_dict.update({file: img.shape})
  count += 1
  if count == 750:
    break


# In[ ]:


df_train = pd.DataFrame(index=file_names)
df_train['width'] = width_list
df_train['height'] = height_list
df_train['ratio'] = ratios_list
df_train


# In[ ]:


avg_shape = round(df_train['width'].mean()), round(df_train['height'].mean())
min_shape = round(df_train['width'].min()), round(df_train['height'].min())
max_shape = round(df_train['width'].max()), round(df_train['height'].max())
print("Average shape : ", (avg_shape))
print("Minimum shape : ", (min_shape))
print("Maximum shape : ", (max_shape))


# In[ ]:


smallest_image = df_train[df_train['width'] == min_shape[0]].index.values[0]
img = cv2.imread(path+'/'+smallest_image)
plt.imshow(img)
plt.title(img.shape, fontdict={'fontsize': 20})


# In[ ]:


largest_image = df_train[df_train['width'] == max_shape[0]].index.values[0]
img = cv2.imread(path+'/'+largest_image)
plt.imshow(img)
plt.title(img.shape, fontdict={'fontsize': 20})


# In[ ]:


x_data = range(0, len(df_train))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(20,5)
fig.suptitle('shapes')
target_size = [400, 400]
target_ratio = [1.0, 1.0]

ax1.plot(x_data, df_train['width'], label=1)
ax1.set_title('weight')
ax2.plot(x_data, df_train['height'], label=2)
ax2.set_title('height')
ax3.plot(x_data, df_train['ratio'], label=3)
ax3.set_title('ratio')
ax1.plot([0, len(df_train)], target_size, color='r', linestyle='-', linewidth=2)
ax2.plot([0, len(df_train)], target_size, color='r', linestyle='-', linewidth=2)
ax3.plot([0, len(df_train)], target_ratio, color='r', linestyle='-', linewidth=2)


# ## **Modifying**

# ### Resize

# In[ ]:


def resize(img, target_size):
  img_resized = cv2.resize(img, (target_size[0], target_size[1]))
  return img_resized


# In[ ]:


img = cv2.imread(path+'/'+largest_image)
img_resized = resize(img, (target_size[0], target_size[1]))
print('Before resizing: ', img.shape[:2])
print('After resizing: ', img_resized.shape[:2])

fig = plt.figure(figsize=(10, 20))
fig.add_subplot(1, 2, 1)
plt.imshow(img)
fig.add_subplot(1, 2, 2)
plt.imshow(img_resized)
plt.show()


# ### Expand

# In[ ]:


def expand(img, target_size):
  if img.shape[0] < target_size[0]:
    w = img.shape[0]
  else:
    w = target_size[0]
  if img.shape[1] < target_size[1]:
    h = img.shape[1]
  else:
    h = target_size[1]
  x_pad = (math.floor((target_size[0] - w)/2), math.ceil((target_size[0] - w)/2))
  y_pad = (math.floor((target_size[1] - h)/2), math.ceil((target_size[1] - h)/2))
  img_expanded = np.stack([np.pad(img[:,:,x], [x_pad, y_pad], mode='constant', constant_values=0) for x in range(3)], axis=2)
  return img_expanded


# In[ ]:


img = cv2.imread(path+'/'+smallest_image)
print('before', img.shape)
img_resized = resize(img, (target_size[0], target_size[1]))
img_expanded = expand(img, (target_size[0], target_size[1]))

fig = plt.figure(figsize=(13, 27))
fig.add_subplot(1, 3, 1)
plt.imshow(img)
fig.add_subplot(1, 3, 2)
plt.imshow(img_resized)
fig.add_subplot(1, 3, 3)
plt.imshow(img_expanded)
plt.show()


# ## **Equalizing all images at (400, 400)**

# In[ ]:


def equalize(path, target_size, target_path):
  img = cv2.imread(path)
  try:
    if (img.shape[0] < target_size[0] and img.shape[1] < target_size[1]) or (
        img.shape[0] == target_size[0] and img.shape[1] < target_size[1]) or (
            img.shape[0] < target_size[0] and img.shape[1] == target_size[1]):
      img_modified = expand(img, target_size)
    elif (img.shape[0] < target_size[0] and img.shape[1] > target_size[1]) or (
        img.shape[0] > target_size[0] and img.shape[1] < target_size[1]):
      img_modified = expand(img, target_size)
      img_modified = resize(img, target_size)
    elif (img.shape[0] > target_size[0] and img.shape[1] > target_size[1]) or (
        img.shape[0] == target_size[0] and img.shape[1] > target_size[1]) or (
            img.shape[0] > target_size[0] and img.shape[1] == target_size[1]):
      img_modified = resize(img, target_size)
    elif (img.shape[0] == target_size[0] and img.shape[1] == target_size[1]):
      img_modified = img
  except: 
    print('ANOTHER CASE HAS BEEN ENCOUNTERED! FOR:', path)

  cv2.imwrite(target_path, img_modified)


# In[ ]:


os.mkdir('/tmp/train_modified')
os.mkdir('/tmp/test_modified')
os.mkdir('/tmp/train_modified/train')
os.mkdir('/tmp/test_modified/test')


# In[ ]:


path = '/kaggle/input/dogs-vs-cats/train/train'
target_path = '/tmp/train_modified/train'

for file in os.listdir(path):
  if '.ipynb_checkpoints' not in file:
    try:
      equalize(path+'/'+file, target_size, target_path+'/'+file)
    except:
      print(f"Error: Couldn't equalize {file}.")


# In[ ]:


count = 0
to_be_deleted = []
for file in os.listdir(target_path):
    img = cv2.imread(f'{target_path}/{file}', 0)
    try:  
        if img.shape != (400, 400):
            print(file, img.shape)
            to_be_deleted.append(f'{target_path}/{file}')
    except:
        print("ERROR FOR: ", file, img.shape)
        pass


# In[ ]:


img = cv2.imread(f'{path}/cat.7904.jpg', 0)
print(img.shape)
equalize(f'{path}/cat.7904.jpg', target_size, target_path+'/'+file)
img_eq = cv2.imread(f'{target_path}/cat.7904.jpg', 0)
print(img_eq.shape)


# In[ ]:


to_be_deleted


# In[ ]:


for file in to_be_deleted:
  os.remove(file)


# In[ ]:


_, _, files = next(os.walk(target_path))
len(files)


# Test

# In[ ]:


path = '/kaggle/input/dogs-vs-cats/test1/test1'
target_path = '/tmp/test_modified/test'

for file in os.listdir(path):
  if '.ipynb_checkpoints' not in file:
    try:
      equalize(path+'/'+file, target_size, target_path+'/'+file)
    except:
      print(f"Error: Couldn't equalize {file}.")


# In[ ]:


count = 0
to_be_deleted = []
for file in os.listdir(target_path):
    img = cv2.imread(f'{target_path}/{file}', 0)
    try:  
        if img.shape != (400, 400):
            print(file, img.shape)
            to_be_deleted.append(f'{target_path}/{file}')
    except:
        print("ERROR FOR: ", file, img.shape)
        pass


# In[ ]:


_, _, files = next(os.walk(target_path))
len(files)


# ## **Model**

# In[ ]:


#part 1 - Building CNN
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers import Convolution2D,MaxPooling2D
import cv2,numpy as np
from keras.models import load_model
import glob

def create_cnn(model_path=None):
    #initialization
    classifier=Sequential()
    
    #Convolution
    classifier.add(Convolution2D(32,3,3,input_shape=(400, 400, 3),activation='relu'))

    #Pooling
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    #Dropout
    classifier.add(Dropout(0.5))
    #Adding 2nd conv. layaer
    classifier.add(Convolution2D(32,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.5))
    
    classifier.add(Convolution2D(64,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.5))
    
    
    classifier.add(Convolution2D(64,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.5))
    
    #Flattening
    classifier.add(Flatten())

    #Full Connected Layers
    classifier.add(Dense(output_dim=64,activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(output_dim=128,activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(output_dim=1,activation='sigmoid'))

    #compliling CNN
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    #Fitting CNN to Images
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        '/tmp/train_modified',
        target_size=(400, 400),
        batch_size=32,
        class_mode='binary')
    
    test_set = test_datagen.flow_from_directory(
        '/tmp/test_modified',
        target_size=(400, 400),
        batch_size=32,
        class_mode='binary')

    classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=35,
        validation_data=test_set,
        validation_steps=2000)

    
    classifier.save('classifier.h5')
    
    if model_path:
        model=load_model('classifier.h5')
    
    return model

if __name__ == "__main__":
    #Opening the image for prediction
    file_path=f'/tmp/test_modified/test/1.jpg'
    im=cv2.resize(cv2.imread(file_path),(400, 400)).astype(np.float32)
    #cv2.imshow('Sample',cv2.resize(cv2.imread(file_path),(640,480)))
    im = np.expand_dims(im, axis=0)
    
    #Checking if model is present in the Directory
    if(glob.glob('*.h5')):
        for filename in glob.glob('*.h5'):
            model=load_model(filename)
            print('Model Loaded')
    else:
        print('Model Not found, Creating a new Model')
        model=create_cnn('classifier.h5')
    
    #Compiling the model and predicting Output    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    out=model.predict(im)
    
    if out[0][0]==1:
        prediction='Dog'
    else:
        prediction='Cat'
    print(prediction)
    

##Saving the model
# serialize model to JSON
from keras.models import model_from_json
classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
   json_file.write(classifier_json)
   
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")

# load json and create model
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifier.h5")
print("Loaded model from disk")


# In[ ]:




