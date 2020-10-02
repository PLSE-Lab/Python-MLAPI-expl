#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install sklearn')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
        
try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv("../input/adult-census-income/adult.csv")

# Check for Null Data
dataset.isnull().sum()


# In[ ]:


dataset = dataset.fillna(np.nan)


# In[ ]:


dataset.dtypes


# In[ ]:


dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
dataset.head(4)


# In[ ]:


dataset["workclass"] = dataset["workclass"].fillna("X")
dataset["occupation"] = dataset["occupation"].fillna("X")
dataset["native.country"] = dataset["native.country"].fillna("United-States")

# Confirm All Missing Data is Handled
dataset.isnull().sum()


# In[ ]:


dataset["sex"] = dataset["sex"].map({"Male": 0, "Female":1})

# Create Married Column - Binary Yes(1) or No(0)
dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
dataset["marital.status"] = dataset["marital.status"].map({"Married":1, "Single":0})
dataset["marital.status"] = dataset["marital.status"].astype(int)


# In[ ]:


numeric_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','marital.status', 'sex']

# Identify Categorical features
cat_features = ['education', 'relationship', 'race', 'native.country']


# In[ ]:


train, test = train_test_split(dataset, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# In[ ]:


feature_columns = []

# numeric cols
for header in numeric_features:
  feature_columns.append(feature_column.numeric_column(header))

feature_columns


# In[ ]:


dataset.head()


# In[ ]:


dataset["workclass"] = dataset["workclass"].replace('?', 'X')
dataset["occupation"] = dataset["occupation"].replace('?', 'X')
dataset.head()


# In[ ]:


dataset.drop(labels=["workclass","occupation"], axis = 1, inplace = True)
print('Dataset with Dropped Labels')
print(dataset.head())


# In[ ]:


for feature in cat_features:
    l = dataset[feature].unique()
#     print(l)
    f = feature_column.categorical_column_with_vocabulary_list(feature,l)
    one_hot = feature_column.indicator_column(f)
    feature_columns.append(one_hot)
    
print(feature_columns)


# In[ ]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('income')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


# In[ ]:


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# In[ ]:


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# In[ ]:


model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=25)


# In[ ]:


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


# In[ ]:





# In[ ]:




