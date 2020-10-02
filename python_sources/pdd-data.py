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
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
train.head()


# ## Test data

# In[ ]:


test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
test.head()


# ## Train shape

# In[ ]:


print("Train data shape : rows :-{0} , columns:- {1}".format(train.shape[0], train.shape[1]))


# In[ ]:


train.info()


# ## Check for any empty records 

# In[ ]:


print('Number of empty records :', train.isnull().any().sum())


# In[ ]:





# In[ ]:


path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'

for i in train['image_id'][:5]:
    print(i)


# 

# # Exploratory Data Analysis

# In[ ]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Sample images

# In[ ]:


image_index = 5
full_path = path + train['image_id'][image_index] + '.jpg'
img = Image.open(full_path)
plt.imshow(img)

print('Image default size', img.size)


# ##  Healthy Images

# In[ ]:


train_healthy = train[train['healthy'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('Healthy')
plt.show()


# ## Scab Images

# In[ ]:


train_healthy = train[train['scab'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('scab')
# plt.show()


# ## Rust Images

# In[ ]:


train_healthy = train[train['rust'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('rust')
plt.show()


#  ## Multiple Diseases Images

# In[ ]:


train_healthy = train[train['multiple_diseases'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('multiple_diseases')
plt.show()


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


data = train[['healthy', 'multiple_diseases', 'rust', 'scab']].sum(axis=0)
data.plot(kind='bar')
plt.title('Frequency count')


# In[ ]:


## pie chart


# In[ ]:


colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

data.plot(kind='pie', colors=colors, title='Data with pie chart', figsize=(10,10)
)


# Note: Found that dataset is imbalance

# * # Data preprocessing

# In[ ]:


from keras.preprocessing import image 
import numpy as np


# In[ ]:


img_size  = 224 # image size during training 


# ## Train data

# In[ ]:


load_features = []

for i in train['image_id']:
    full_path = path + i + '.jpg'
    img = image.load_img(full_path, target_size=(img_size,img_size,3), color_mode = "rgb")
    img = image.img_to_array(img)
    load_features.append(img)


# ## Features

# In[ ]:


X = np.asarray(load_features)


# In[ ]:


print('Shape of features:',X.shape)


# ## Labels data

# In[ ]:


train.head()
y = train.iloc[:, 1:]
y = y.to_numpy()

y.shape


# In[ ]:


y[:5]


# ## Distribution

# In[ ]:


y.sum(axis=0)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train ,x_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=42)


# In[ ]:


print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)


# ## Standerdization

# In[ ]:


x_train_final = x_train /255
x_train_final.shape


# ## X test final

# In[ ]:


x_test_final = x_test /255
x_test_final.shape


# ## Y train

# In[ ]:


y_train_final = y_train
y_train_final.shape


# ## Y test

# In[ ]:


y_test_final = y_test
y_test_final.shape


# In[ ]:





# ## Model evaluation

# In[ ]:




