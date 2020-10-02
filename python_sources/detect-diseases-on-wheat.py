#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import required libraries
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import re
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import tensorflow as tf 
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


# In[ ]:


data_dir = "../input/split-data-into-directories/train_val/"


# In[ ]:


# get to know whats inside the data
os.listdir(data_dir)


# In[ ]:


# initialize image sizes and parameters
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32


# In[ ]:


os.listdir(data_dir + "train")


# # process the data with tf.data 
# 

# In[ ]:


# a function to show the image batch
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')


# In[ ]:


# get the number of images
train_image_count = len(list(Path(data_dir + "train").glob('*/*')))
val_image_count = len(list(Path(data_dir + "val").glob('*/*')))
print(train_image_count)
val_image_count


# In[ ]:


# train data
train_list_ds = tf.data.Dataset.list_files(str(data_dir + "train/" +'*/*'))
# validation data
val_list_ds = tf.data.Dataset.list_files(str(data_dir + "val/" +'*/*'))


# In[ ]:


# images class names
CLASS_NAMES = np.array([item.name for item in Path(data_dir + "train").glob('*')])
CLASS_NAMES


# In[ ]:


# check the images file paths
for f in train_list_ds.take(5):
  print(f.numpy())


# In[ ]:


# read image 
# a function to read data
def read_image(img_path):
    img_loader = tf.io.read_file(img_path)
    img_decoder = tf.image.decode_jpeg(img_loader, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img_decoder, tf.float32)
    img = tf.image.resize(img, [WIDTH, HEIGHT])
    # img = img/255.0 # rescale images
    return img


# In[ ]:


# To get labels
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


# In[ ]:


# process paths
def process_path(file_path):
    label = get_label(file_path)
    label = tf.cast(label, tf.int32)
    img = read_image(file_path)
    return img, label


# In[ ]:


# using tensorflow dataset.map to create a dataset of image, label pairs
train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=-1)
val_labeled_ds = val_list_ds.map(process_path, num_parallel_calls=-1)


# In[ ]:


for image, label in val_labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


# In[ ]:


# prepare data for training 
def prepare_for_training(ds, shuffle_buffer_size=100, training=True):
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  if training:
    ds = ds.repeat()
    print("repeating")
  else:
    pass
  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=-1)

  return ds


# In[ ]:


train_ds = prepare_for_training(train_labeled_ds)
val_ds = prepare_for_training(val_labeled_ds, training=False)

# get some data
image_batch, label_batch = next(iter(train_ds))


# In[ ]:


show_batch(image_batch.numpy(), label_batch.numpy())


# In[ ]:


# prepare the test data
test_list = "../input/cgiar-computer-vision-for-crop-disease/ICLR/test/"


# In[ ]:


test_list_ds = tf.data.Dataset.list_files(str(test_list +'*/*'))


# In[ ]:


# test list file names
for f in test_list_ds.take(5):
  print(f.numpy())


# In[ ]:


# process test
def process_test(image_path):
    img = read_image(image_path)
    return img, image_path


# In[ ]:


test_ds = test_list_ds.map(process_test, num_parallel_calls=-1)


# In[ ]:


for i, j in test_ds.take(1):
    print(i.numpy().shape)
    print(j.numpy())
    plt.imshow(i.numpy())


# # create callbacks

# In[ ]:


# temsorboard
log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early stopping
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# model checkpoints
checkpoint_path =  "model/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp.ckpt"
model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_categorical_accuracy',
            mode='max')


# In[ ]:


callbacks = [tensorboard, es]


# In[ ]:


# define class weights
leaf_rust_count = len(os.listdir(data_dir + "train/leaf_rust"))
stem_rust_count = len(os.listdir(data_dir + "train/stem_rust"))
healthy_wheat_count =len(os.listdir(data_dir + "train/healthy_wheat"))
total = leaf_rust_count + stem_rust_count + healthy_wheat_count

leaf_rust_weight = (1/leaf_rust_count) * (total) / 3.0
stem_rust_weight = (1/stem_rust_count) * (total) / 3.0
healthy_wheat_weight = (1/healthy_wheat_count) * (total) / 3.0

class_weight = {0:leaf_rust_weight, 1:stem_rust_weight, 2:healthy_wheat_weight}
print(class_weight)


# #  create a simple model

# In[ ]:


def simple_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32 ,(3,3), activation="relu", 
                padding="same", input_shape=(224,224, 3)))
    model.add(tf.keras.layers.MaxPool2D(2,2, padding="same"))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
                    loss=tf.keras.losses.categorical_crossentropy, 
                    metrics = [tf.keras.metrics.categorical_accuracy])
    return model 


# In[ ]:


steps_per_epoch = train_image_count // BATCH_SIZE
validation_steps = val_image_count // BATCH_SIZE
print(steps_per_epoch)
validation_steps


# In[ ]:


# fit the model
model = simple_model()


# In[ ]:


model.fit(train_ds, epochs=1000, steps_per_epoch=steps_per_epoch, 
                validation_data=val_ds, validation_steps=validation_steps,
                callbacks=callbacks, class_weight=class_weight)


# # Make prediction for the test dataset

# In[ ]:


# test count
test_count = len(list(Path(test_list).glob('*/*')))
test_count


# In[ ]:


names = []
preds = []


# In[ ]:


for i, j in tqdm(test_ds):
    i = i.numpy()[np.newaxis, :] # add a new dimension
    prediction = model.predict_proba(i) # make predictions
    preds.append(prediction) 
    
    # use regular expressions to extract the name of image
    name = j.numpy()
    name = re.sub("[^A-Z0-9]", "", str(name))
    name = name.replace("JPG", "")
    name = name.replace("PNG", "")
    name = name.replace("JPEG", "")
    name = name.replace("JFIF", "")
    names.append(name)
    # break


# # Create a submission file

# In[ ]:


# create a dummy dataset
leaf_rust = pd.Series(range(610), name="leaf_rust", dtype=np.float32)
stem_rust = pd.Series(range(610), name="stem_rust", dtype=np.float32)
healthy_wheat = pd.Series(range(610), name="healthy_wheat", dtype=np.float32)


# In[ ]:


sub = pd.concat([leaf_rust, stem_rust, healthy_wheat], axis=1)


# In[ ]:


sub.shape


# In[ ]:


# append real predictions to the dataset
for i in tqdm(range(0 ,len(preds))):
    sub.loc[i] = preds[i]
    # break


# In[ ]:


sub.head()


# In[ ]:


sub["ID"] = names


# In[ ]:


sub.tail()


# In[ ]:


cols = sub.columns.tolist()


# In[ ]:


cols = cols[-1:] + cols[:-1]
sub = sub[cols]


# In[ ]:


sub.head()


# In[ ]:


# write to csv
sub.to_csv("sub.csv", index=False)

