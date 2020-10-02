#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().system('pip install py7zr')
from keras.preprocessing.image import load_img,img_to_array
from py7zr import unpack_7zarchive
import shutil
import os
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


# In[ ]:


shutil.unpack_archive('/kaggle/input/cifar-10/train.7z', '/kaggle/working')


# In[ ]:



train_dir = os.listdir("./train");
train_dir_len = len(train_dir)
print(".\\train:\t",train_dir_len)
print("files:\t\t",train_dir[:3])


# In[ ]:


train_labels = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv',dtype=str)
train_images = pd.DataFrame(columns = ['id','label','path'],dtype=str)
test_labels = pd.read_csv('/kaggle/input/cifar-10/sampleSubmission.csv')
train_labels.info()


# In[ ]:


path_base = '/kaggle/working/train/'

for index in range(0,train_dir_len):
    path = path_base + str(index+1)+'.png'
    if os.path.exists(path):
        train_images = train_images.append([{ 'id': str(train_labels['id'].iloc[index]),'path': path, 'label':train_labels['label'].iloc[index]}])
        
train_images.head(2)


# In[ ]:


train_images.head(2)


# In[ ]:


display_groupby = train_images.groupby(['label']).count()
display_groupby.head(10)


# In[ ]:


class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for name in  class_names:
    index = class_names.index(name)
    train_images.loc[train_images.label==name,'label'] = str(index)
        
display_groupby = train_images.groupby(['label']).count()
display_groupby.head(10)


# In[ ]:


path_base = '/kaggle/working/train'
batch_size = 64
train_data_generator = ImageDataGenerator(
            rescale=1./255.,
            validation_split=0.2,
            horizontal_flip=True
            )
train_generator = train_data_generator.flow_from_dataframe(dataframe=train_images,
            directory="./train/",
            x_col="path",
            y_col="label",
            subset="training",
            batch_size=batch_size,
            shuffle=True,
            target_size=(32,32),
            class_mode="categorical")


# In[ ]:


num_classes  = 10


# In[ ]:


validation_generator = train_data_generator.flow_from_dataframe(dataframe=train_images,
            directory="./train/",
            x_col="path",
            y_col="label",
            subset="validation",
            batch_size=batch_size,
            shuffle=True,
            target_size=(32,32),
            class_mode="categorical")


# In[ ]:


train_size = len(train_generator.filenames)
validation_size = len(validation_generator.filenames)
print('validation_size:\t',validation_size)
print('train_size:\t\t',train_size)


# In[ ]:


index = 0    
fig = plt.figure(figsize = (16,10))
for item in train_images.values[:20]:
    index += 1
    plt.subplot(5, 5, index)
    test_path = item[2]
    test_image = load_img(test_path, target_size=(32,32))
    plt.imshow(test_image)
    plt.colorbar()
    plt.grid(False)
    plt.axis("off")
    plt.title(class_names[int(item[1])])
plt.show()


# In[ ]:


keras.backend.clear_session()

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32,  3)))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (2, 2), activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(1, 1))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (2, 2), activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(1, 1))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
          optimizer=keras.optimizers.RMSprop(lr=0.001, decay = 1e-3, momentum = 0.3),
          metrics=['accuracy'])
    
model.input 


# In[ ]:



history = model.fit(train_generator, 
                    steps_per_epoch=(train_size//batch_size),
                    epochs= 5,
                    validation_data=validation_generator,
                   validation_steps=(validation_size//batch_size)
                   )


# In[ ]:


test_labels.head(2)


# In[ ]:


if os.path.exists("./test"):
    shutil.rmtree("./test")
if os.path.exists("./train"):
    shutil.rmtree("./train")
if not os.path.exists("./data"):
    os.mkdir("./data")

shutil.unpack_archive('/kaggle/input/cifar-10/test.7z', '/kaggle/working/data')


# In[ ]:


test_dir = os.listdir("./data/test");
test_dir_len = len(test_dir)
print('min:\t',min(test_dir))
print('max:\t',max(test_dir))
print(".\\test:\t",test_dir_len)
print("files:\t\t",test_dir[:3])


# In[ ]:


test_data_generator = ImageDataGenerator(rescale=1./255.)
test_generator = test_data_generator.flow_from_directory(directory='/kaggle/working/data',
            batch_size=batch_size,
            shuffle=False,color_mode='rgb',
            target_size=(32,32),
            class_mode=None)


# In[ ]:


predict_test = model.predict_generator(test_generator)


# In[ ]:


predict_generator = np.argmax(predict_test, axis=1)
print(class_names)
predict_generator[:2],[class_names[int(i)] for i in predict_generator[:2]]


# In[ ]:


submission = pd.DataFrame(columns = ['id','label'],dtype=str)
submission["label"] = [class_names[int(i)] for i in predict_generator]
submission["id"] = [ (''.join(filter(str.isdigit, name ))) for name in test_generator.filenames]
submission.head(101)


# In[ ]:


submission.values[50:100]


# In[ ]:


index = 0    
fig = plt.figure(figsize = (16,10))
for item in submission.values[50:70]:
    index += 1
    plt.subplot(5, 5, index)
    test_path = '/kaggle/working/data/test/'+item[0]+'.png'
    test_image = load_img(test_path, target_size=(32,32))
    plt.imshow(test_image)
    plt.colorbar()
    plt.grid(False)
    plt.axis("off")
    plt.title(item[1])
plt.show()


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:


shutil.rmtree("./data")

