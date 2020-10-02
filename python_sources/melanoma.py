#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator


# In[ ]:


train_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'


# In[ ]:


df_train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
df_test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train['image_num'] = df_train.apply(lambda row: train_dir + row['image_name'] + '.jpg', axis=1)


# In[ ]:


df_test['image_num'] = df_test.apply(lambda row: test_dir + row['image_name'] + '.jpg', axis=1)


# In[ ]:


datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)


# In[ ]:


train_generator = datagen.flow_from_dataframe(
dataframe=df_train,
x_col="image_num",
y_col="target",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))


# In[ ]:


valid_generator = datagen.flow_from_dataframe(
dataframe=df_train,
x_col="image_num",
y_col="target",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))

