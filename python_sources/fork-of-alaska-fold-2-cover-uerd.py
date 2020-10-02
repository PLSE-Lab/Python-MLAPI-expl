#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, cv2
import tensorflow as tf
import albumentations as A
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from glob import glob
import random
# import cv2
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API


# In[ ]:


SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(SEED)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# define the class that we need\nclass_1 = 'Cover'\nclass_2 = 'UERD'\n\ndataset = []\n\nfor label, kind in enumerate([class_1, class_2]):\n    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):\n        dataset.append({\n            'kind': kind,\n            'image_name': path.split('/')[-1],\n            'label': label\n        })\n\nrandom.shuffle(dataset)\ndataset = pd.DataFrame(dataset)\n\ngkf = GroupKFold(n_splits=5)\n\ndataset.loc[:, 'fold'] = 0\nfor fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):\n    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number")


# In[ ]:


print(dataset.shape)
dataset.kind.unique()


# In[ ]:


fold_number = 2

# df[(df['Salary_in_1000']>=100) & (df['Age']<60) & df['FT_Team'].str.startswith('S')][['Name','Age','Salary_in_1000']]

# train_data = dataset[dataset['fold'] != 4].reset_index(drop=True)
valid_data = dataset[(dataset['fold'] == fold_number)].reset_index(drop=True)
valid_data.head()


# In[ ]:


# valid_data = valid_data[(valid_data['kind'] == class_1)]
valid_data.kind.value_counts()


# In[ ]:


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    #value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[ ]:


def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0)
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
        ], p=1.0)

train_transform = get_train_transforms()
valid_transform = get_valid_transforms()


# In[ ]:


def onehot(size, target):
    vec = np.zeros(size, dtype=np.float32)
    vec[target] = 1.
    return vec


# In[ ]:


def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# In[ ]:


valid_paths = []

for i in range(len(valid_data)):
    kind = valid_data['kind'].loc[i]
    img_id = valid_data['image_name'].loc[i]
    valid_paths.append(f'../input/alaska2-image-steganalysis/{kind}/{img_id}')
valid_paths[0:5]


# In[ ]:


valid_data['path'] = valid_paths
valid_data.head()


# In[ ]:


valid_data.loc[1]['label'], valid_data.loc[1]['kind'], valid_data.loc[1]['image_name'], valid_data.loc[1]['path']


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# !mkdir valid\n# !mkdir valid/fold_0_1\n\nSIZE = 2000\nCT = valid_data.shape[0]//SIZE + int(valid_data.shape[0]%SIZE!=0)\nfor j in range(CT):\n    print()\n    print('Writing TFRecord %i of %i...'%(j,CT))\n    CT2 = min(SIZE, valid_data.shape[0]-j*SIZE)\n    with tf.io.TFRecordWriter('coverUERDValid%.2i-%i.tfrec'%(j,CT2)) as writer:\n        for k in range(CT2):\n            img = cv2.imread(valid_data.loc[k]['path'])\n            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)\n            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()\n            \n            one_hot_class = onehot(4, valid_data.loc[k]['label'])\n            example = serialize_example(\n                img,\n                valid_data.loc[k]['label'],\n            )\n            writer.write(example)\n            if k%1000==0: print(k,', ',end='')")


# In[ ]:


path = 'coverUERDValid00-2000.tfrec'

def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),  
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.reshape(image, [512, 512, 3])
    class_num = example['label']
    
    return image, class_num


# In[ ]:


dataset4 = tf.data.TFRecordDataset(path)
dataset4 = dataset4.map(read_tfrecord)
# dataset4 = dataset4.shuffle(300)
a = None
count = 0
for tensor in dataset4:
    count += 1
    print(count)
    if count == 2:
        a = tensor[0]
        print(tensor)
        break


# In[ ]:


plt.imshow(a.numpy())


# In[ ]:




