#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import tensorflow as tf


# In[ ]:


import pathlib


# In[ ]:


os.getcwd()


# In[ ]:


train_path = '../input/chest-xray-pneumonia/chest_xray/train/'
val_path = '../input/chest-xray-pneumonia/chest_xray/val/'
test_path = '../input/chest-xray-pneumonia/chest_xray/test/'


# In[ ]:


train_data_dir = pathlib.Path(train_path)
test_data_dir = pathlib.Path(test_path)
val_data_dir = pathlib.Path(val_path)
train_data_dir


# In[ ]:


train_image_count = len(list(train_data_dir.glob('*/*.jpeg')))
train_image_count


# In[ ]:


test_image_count = len(list(test_data_dir.glob('*/*.jpeg')))
test_image_count


# In[ ]:


val_image_count = len(list(val_data_dir.glob('*/*.jpeg')))
val_image_count


# In[ ]:


CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
CLASS_NAMES


# In[ ]:


BATCH_SIZE = 32
IMAGE_SIZE = 224
STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)


# In[ ]:


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,validation_split=0.2)


# In[ ]:


train_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                     class_mode='binary',
                                                     subset='training')


val_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                     class_mode='binary',
                                                  subset='validation')
    


# In[ ]:


test_data_gen = image_generator.flow_from_directory(directory=str(test_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                     class_mode='binary',
                                                     )


# In[ ]:


train_data_gen.class_indices


# In[ ]:


images,labels = train_data_gen.next()
print(images.shape)
print(labels)


# In[ ]:


IMAGE_SHAPE = (224,224,3)


# # USING TPU

# In[ ]:


# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# In[ ]:





# In[ ]:


# # Create the base model from the pre-trained model MobileNet V2
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,include_top=False,weights='imagenet')
# base_model.trainable = False


# In[ ]:


# base_model.summary()


# In[ ]:


IMAGE_SHAPE = (224,224,3)


# In[ ]:


# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,include_top=False,weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1)
    ])
    initial_epochs = 10
    base_learning_rate = 0.001
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate),metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(train_data_gen,
                    epochs=initial_epochs,
                    validation_data=val_data_gen)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # IGNORE ALL  FROM HERE

# In[ ]:


train_path = '../input/chest-xray-pneumonia/chest_xray/train/'
val_path = '../input/chest-xray-pneumonia/chest_xray/val/'
test_path = '../input/chest-xray-pneumonia/chest_xray/test/'


# In[ ]:


train_data_dir = pathlib.Path(train_path)
test_data_dir = pathlib.Path(test_path)
val_data_dir = pathlib.Path(val_path)
train_data_dir


# In[ ]:


train_list_ds = tf.data.Dataset.list_files(str(train_data_dir/'*/*'))
val_list_ds = tf.data.Dataset.list_files(str(val_data_dir/'*/*'))
test_list_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*'))


# In[ ]:


CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES


# In[ ]:


for i in train_list_ds:
    print(i)
    img = tf.io.read_file(i)
    img = tf.image.decode_jpeg(img, channels=3)
    print(img.shape)
    break
parts = tf.strings.split(i, os.path.sep)
print(parts[-2])
print(parts[-1])
print(parts[-2]==CLASS_NAMES)


# ## Decode Images

# In[ ]:


IMAGE_SIZE=224
BATCH_SIZE=32


# In[ ]:


def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES


# In[ ]:


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])


# In[ ]:


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[ ]:


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_labeled_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_labeled_ds = test_list_ds.map(process_path,num_parallel_calls=AUTOTUNE)


# In[ ]:


for image, label in train_labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


# In[ ]:


train_ds = train_labeled_ds.shuffle(1000).batch(BATCH_SIZE)
val_ds = val_labeled_ds.shuffle(1000).batch(16)
test_ds = test_labeled_ds.shuffle(1000).batch(BATCH_SIZE)


# In[ ]:


image_batch, label_batch = next(iter(train_ds))


# In[ ]:


label_batch


# In[ ]:


label_batch[0]


# In[ ]:





# In[ ]:


image_batch, label_batch = next(iter(val_ds))


# In[ ]:


image_batch.shape


# In[ ]:


arr = label_batch.numpy()
arr


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(16):
        ax = plt.subplot(4,4,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
        

show_batch(image_batch.numpy(),label_batch.numpy())


# In[ ]:


arr[0]


# In[ ]:


print(arr[0]==0)
print(arr[0]==1)


# In[ ]:


print(CLASS_NAMES[arr[0]==1])
print(CLASS_NAMES[arr[0]==1][0])
print(CLASS_NAMES[arr[0]==1][0].title())


# In[ ]:


image_batch[0].numpy().max()


# In[ ]:


IMAGE_SHAPE = (IMAGE_SIZE,IMAGE_SIZE,3)
IMAGE_SHAPE


# In[ ]:





# # TRY DIFF. MODELS

# In[ ]:


# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,include_top=False,weights='imagenet')


# In[ ]:


base_model.trainable = False


# In[ ]:


base_model.summary()


# In[ ]:


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax')
])


# In[ ]:


model.summary()


# In[ ]:


base_learning_rate = 0.001
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


initial_epochs = 10
#validation_steps=20

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)


# ### Fine Tunning

# In[ ]:


base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False


# In[ ]:


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:





# In[ ]:


# b = 0 
# for i,l in train_ds:
#     b+=1
# print(b,STEPS_PER_EPOCH)


# In[ ]:


initial_epochs = 10
#validation_steps=20

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)


# In[ ]:


model.evaluate(test_ds)


# In[ ]:


image,label = next(iter(test_ds))


# In[ ]:


model.evaluate(image,label)


# In[ ]:


predict = model.predict(image)


# In[ ]:


predict


# In[ ]:


label


# In[ ]:





# In[ ]:


model1 = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2,activation='softmax')
])

model1.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model1.summary()


# In[ ]:


initial_epochs = 5
#validation_steps=20

history = model1.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)


# In[ ]:





# ## VGG16 Model

# In[ ]:


vgg_base_model = tf.keras.applications.VGG16(input_shape=IMAGE_SHAPE,include_top=False,weights='imagenet')


# In[ ]:


vgg_base_model.trainable = False


# In[ ]:


vgg_base_model.summary()


# In[ ]:


vgg_model = tf.keras.models.Sequential([
    vgg_base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2,activation='softmax')
])


# In[ ]:


vgg_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


initial_epochs = 5

vgg_model.fit(train_ds,
                     epochs=initial_epochs,
                     validation_data=val_ds)


# In[ ]:


len(vgg_base_model.layers)


# In[ ]:


vgg_base_model.trainable = True


# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(vgg_base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 12

# Freeze all the layers before the `fine_tune_at` layer
for layer in vgg_base_model.layers[:fine_tune_at]:
    layer.trainable =  False


# In[ ]:


vgg_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

vgg_model.summary()


# In[ ]:


fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = vgg_model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch =  5,
                         validation_data=val_ds)


# In[ ]:




