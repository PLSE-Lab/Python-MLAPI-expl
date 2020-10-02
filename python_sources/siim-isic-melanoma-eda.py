#!/usr/bin/env python
# coding: utf-8

# # SIIM - ISIC - Melanoma Classification

# Melanoma is the deadliest form of skin cancer. The goal of this competition is to develop a model which can classifiy benign and malignant skin lesions. Let's first understand some basic statistics and charachteristics about Melanoma through this [video](https://www.youtube.com/watch?v=DS9eIhcuWEM) 
# [![video](https://img.youtube.com/vi/DS9eIhcuWEM/0.jpg)](https://www.youtube.com/watch?v=DS9eIhcuWEM)

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path = '/kaggle/input/siim-isic-melanoma-classification'
df_train = pd.read_csv(f"{path}/train.csv")
df_test = pd.read_csv(f"{path}/test.csv")


# ## EDA

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly


# In[ ]:


print(f'Train/Test ratio: {df_train.shape[0]/df_test.shape[0]}')


# Train set is 3 times as bigger as test set

# In[ ]:


print(f"Train Male-female ratio: {df_train.loc[df_train['sex']=='male'].shape[0]/df_train.loc[df_train['sex']=='female'].shape[0]}")


# In[ ]:


print(f"Test Male-female ratio: {df_test.loc[df_test['sex']=='male'].shape[0]/df_test.loc[df_test['sex']=='female'].shape[0]}")


# We have slightly more males in test than in train but I think that is fine because train set is also 3 times bigger than test set

# In[ ]:


num_malignant_males = df_train.loc[(df_train['target']==1) & (df_train['sex']=='male'), :].shape[0]
num_malignant_females = df_train.loc[(df_train['target']==1) & (df_train['sex']=='female'), :].shape[0]
print(f"Male ratio in Malignant: {num_malignant_males/(num_malignant_males + num_malignant_females)}")

num_benign_males = df_train.loc[(df_train['target']==0) & (df_train['sex']=='male'), :].shape[0]
num_benign_females = df_train.loc[(df_train['target']==0) & (df_train['sex']=='female'), :].shape[0]
print(f"Male ratio in Benign: {num_benign_males/(num_benign_males + num_benign_females)}")


# So number of males in malignant cases are more

# In[ ]:


fig, ax = plt.subplots(figsize=(6,6))
sns.countplot(x='sex', data=df_train.loc[df_train['target']==1], ax=ax, palette={'male': '#5db1e4','female': '#fb9ed6'})
ax.set_title('Sex Distribution in Malignant')


# In[ ]:


fig, ax = plt.subplots(figsize=(6,6))
sns.countplot(x='sex', 
              data=df_train.loc[df_train['target']==0], 
              ax=ax, 
              palette={'male': '#5db1e4','female': '#fb9ed6'})
ax.set_title('Sex Distribution in Benign')


# In[ ]:


fig, ax = plt.subplots(figsize=(12,9))
sns.countplot(x='age_approx', 
              hue='sex', 
              data=df_train.loc[df_train['target']==1], 
              ax=ax, 
              palette={'male': '#5db1e4','female': '#fb9ed6'})
ax.set_title('Age-Sex Distribution in Malignant cases')


# In[ ]:


fig, ax = plt.subplots(figsize=(12,9))
sns.countplot(x='age_approx', 
              hue='sex', 
              data=df_train.loc[df_train['target']==0], 
              ax=ax, 
              palette={'male': '#5db1e4','female': '#fb9ed6'})
ax.set_title('Age-Sex Distribution in Benign cases')


# Not accurately but it seems that men are more likely to get Melanoma especially for age group of above 60 number of malignant cases in men are marginally high

# In[ ]:


fig, ax = plt.subplots(figsize=(12,9))
sns.countplot(x='anatom_site_general_challenge', 
              hue='sex', 
              data=df_train.loc[df_train['target']==1], 
              ax=ax, 
              palette={'male': '#5db1e4','female': '#fb9ed6'})
ax.set_title('Site-Sex Distribution in Malignant cases')


# In[ ]:


fig, ax = plt.subplots(figsize=(12,9))
sns.countplot(x='anatom_site_general_challenge', 
              hue='sex', 
              data=df_train.loc[df_train['target']==0], 
              ax=ax, 
              palette={'male': '#5db1e4','female': '#fb9ed6'})
ax.set_title('Site-Sex Distribution in Benign cases')


# So in general torso is where people get skin patches. 

# In[ ]:


fig, ax = plt.subplots(figsize=(6,6))
sns.countplot(x='diagnosis', 
              data=df_train.loc[df_train['target']==1], 
              ax=ax)
ax.set_title('Diagnosis Distribution in Malignant cases')


# In[ ]:


fig, ax = plt.subplots(figsize=(24,6))
sns.countplot(x='diagnosis', 
              data=df_train.loc[df_train['target']==0], 
              ax=ax)
ax.set_title('Diagnosis Distribution in Benign cases')


# It will be interesting to see Nevus, seborrheic keratosis etc and malanoma if we can find some clue

# ### Benign images

# In[ ]:


import os

# benign images only
images = df_train.loc[df_train['target']==0, 'image_name'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(9)]

# Location of the image dir
img_dir = path +'/jpeg/train'

print('Display Random Benign Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.title(str(df_train.loc[df_train['image_name']==random_images[i].split('.')[0], 'benign_malignant'].item()))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   


# ### Malignant images

# In[ ]:


import os

# malignant images only
images = df_train.loc[df_train['target']==1, 'image_name'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(9)]

# Location of the image dir
img_dir = path +'/jpeg/train'

print('Display Random Malignant Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.title(str(df_train.loc[df_train['image_name']==random_images[i].split('.')[0], 'benign_malignant'].item()))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   


# Lets see types of benign

# In[ ]:


import os

# benign-unknown images only
images = df_train.loc[(df_train['target']==0) & (df_train['diagnosis']=='unknown'), 'image_name'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(9)]

# Location of the image dir
img_dir = path +'/jpeg/train'

print('Display Random Bening - unknown diagnosed Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.title('benign-' + str(df_train.loc[df_train['image_name']==random_images[i].split('.')[0], 'diagnosis'].item()))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   


# In[ ]:


import os

# benign nevus images only
images = df_train.loc[(df_train['target']==0) & (df_train['diagnosis']=='nevus'), 'image_name'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(9)]

# Location of the image dir
img_dir = path +'/jpeg/train'

print('Display Random  Benign - nevus diagnosed Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.title('benign-' + str(df_train.loc[df_train['image_name']==random_images[i].split('.')[0], 'diagnosis'].item()))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   


# The video said if the the shape is not symmetric and if the circumference is rugged it might be mealnoma. We might make some feature around this but lets keep this for later

# ## Modelling

# In[ ]:


IMG_SIZE = 100


# In[ ]:


df_train.head()


# In[ ]:


from tqdm import tqdm


# In[ ]:


import cv2
X = []
y = []
for img in tqdm(os.listdir(f'{path}/jpeg/train')):
    img_array = cv2.imread(os.path.join(f'{path}/jpeg/train', img))
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    X.append(img_array.flatten())
    y.append(df_train.loc[df_train['image_name'] == img.split('.')[0], 'target'])


# In[ ]:


train = pd.DataFrame(data=X)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import os
import tensorflow as tf


# In[ ]:


tfrecord_location = f'{path}/tfrecords'
filename = os.path.join(tfrecord_location)


# In[ ]:


dataset = tf.data.TFRecordDataset(filename)


# In[ ]:


filenames = f'{path}/tfrecords/train00-2071.tfrec'
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset


# In[ ]:


for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)


# In[ ]:


tf.io.parse_single_example(example_proto, feature_description)


# In[ ]:


def read_tfrecord(serialized_example):
    feature_description = { 'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)
           }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    image = tf.io.parse_tensor(example['image'], out_type = float)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)
    
    return image, example['label']


# In[ ]:


tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_location)
parsed_dataset = tfrecord_dataset.map(read_tfrecord)

plt.figure(figsize=(10,10))
for i, data in enumerate(parsed_dataset.take(9)):
    img = tf.keras.preprocessing.image.array_to_img(data[0])
    plt.subplot(3,3,i+1)
    plt.imshow(img)
plt.show()


# In[ ]:


print(tf.__version__)


# In[ ]:


import glob
reader = tf.TFRecordReader()
filenames = glob.glob('/tfrecords/train*')
filename_queue = tf.train.string_input_producer(
   filenames)
_, serialized_example = reader.read(filename_queue)
feature_set = { 'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)
           }
           
features = tf.parse_single_example( serialized_example, features= feature_set )
label = features['label']
 
with tf.Session() as sess:
    print(sess.run([image,label]))


# In[ ]:


def decode(serialized_example):
  """
  Parses an image and label from the given `serialized_example`.
  It is used as a map function for `dataset.map`
  """
  IMAGE_SIZE = 28
  IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
  
  # 1. define a parser
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # 2. Convert the data
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int32)
  # 3. reshape
  image.set_shape((IMAGE_PIXELS))
  return image, label


# In[ ]:


dataset = dataset.map(decode)


# 

# Inspiration
# 
# https://www.kaggle.com/parulpandey/melanoma-classification-eda-starter

# In[ ]:




