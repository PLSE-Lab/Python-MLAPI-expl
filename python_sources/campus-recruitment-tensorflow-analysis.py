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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # We would like to predict Placement

# In[ ]:


import tensorflow as tf


# In[ ]:


#Data Exploration
filepath = '/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
df = pd.read_csv(filepath, nrows = 1000)


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


len(df)


# In[ ]:


ds_raw = tf.data.experimental.make_csv_dataset(filepath, batch_size = 5, label_name = 'status', num_epochs = 1, shuffle = False)


# In[ ]:


for features, labels in ds_raw.take(1):
    for key, values in features.items():
        print(f'{key:20}: {values}')
    print(labels.numpy())


# In[ ]:


#First let's encode the status label
ds = ds_raw.map(lambda features, label: (features, tf.map_fn(lambda x: 1 if x == 'Placed' else 0, label, dtype = tf.int32)))


# In[ ]:


for features, labels in ds.take(1):
    for key, values in features.items():
        print(f'{key:20}: {values}')
    print(labels.numpy())


# In[ ]:


#Before we go into the next step, we would like to deal with NaN values, and only salary feature has lots of NaNs. 
#All NaNs were loaded as 0 by default in tf.data.experimental.make_csv_dataset
#we would just drop salary feature because of the amount of NaNs


# In[ ]:


def popping(input_dict, feature_name):
    input_dict.pop(feature_name)
    return input_dict


# In[ ]:


ds = ds.map(lambda features, label: (popping(features, 'salary'), label))
ds = ds.map(lambda features, label: (popping(features, 'sl_no'), label)) #drop Serial Number, too


# In[ ]:


#Next we would like to get the numeric columns apart from categorical columns
column_dtypes = {}
for key, value in ds.element_spec[0].items():
    column_dtypes[key] = value.dtype


# In[ ]:


column_dtypes


# In[ ]:


numeric_features = [column for column, dtype in column_dtypes.items() if dtype in [tf.int32, tf.float32]]
categorical_features = [column for column, dtype in column_dtypes.items() if dtype == tf.string]


# In[ ]:


#For numeric features, we need to standardize, here we need to go back to the original Pandas dataframe and grab the mean and std
metadata = df.drop(['salary', 'sl_no'], axis = 1).describe().T
metadata


# In[ ]:


#Next we pack all the numeric features
class PackNumericFeatures():
    def __init__(self, names):
        self.names = names
        
    def __call__(self, features, label):
        numeric = [features.pop(name) for name in self.names]
        numeric = [tf.cast(item, tf.float32) for item in numeric]
        numeric = tf.stack(numeric, -1)
        features['numeric'] = numeric
        
        return features, label


# In[ ]:


ds = ds.map(PackNumericFeatures(numeric_features))


# In[ ]:


for features, labels in ds.take(1):
    for key, values in features.items():
        print(f'{key:20}: {values}')
    print(labels.numpy())


# In[ ]:


#Now we can actually perform train test split
print(f'There are {len(df)} records.')

#The reason we only take 40 below is because each batch contains 5 records
ds_train = ds.take(40)
ds_test = ds.skip(40)


# In[ ]:


ds_test.reduce(0, lambda x, _: x + 1).numpy()


# In[ ]:


#Next, we need a Standardization function for the numeric columns
def Standardize(tensor, mean, std):
    return (tensor - mean)/std

mean = metadata['mean'].to_numpy()
std = metadata['std'].to_numpy()


# In[ ]:


mean


# In[ ]:


import functools

Standardize_partial = functools.partial(Standardize, mean = mean, std = std)


# # Next we would build feature columns

# In[ ]:


#for numeric columns
numeric_column = tf.feature_column.numeric_column('numeric', shape = (len(numeric_features), ), normalizer_fn = Standardize_partial)
numeric_columns = [numeric_column]


# In[ ]:


#for categorical columns
print(categorical_features)

categories = {'gender': ['M', 'F'],
             'ssc_b': ['Central', 'Others'],
              'hsc_b': ['Central', 'Others'],
              'hsc_s': ['Commerce', 'Science', 'Arts'],
              'degree_t': ['Comm&Mgmt', 'Sci&Tech', 'Others'],
              'workex': ['Yes', 'No'],
              'specialisation': ['Mkt&Fin', 'Mkt&HR']
             }

categorical_columns = [tf.feature_column.categorical_column_with_vocabulary_list(key, values) for key, values in categories.items()]
categorical_columns = [tf.feature_column.indicator_column(column) for column in categorical_columns]


# In[ ]:


feature_columns = numeric_columns + categorical_columns


# # Next we would train the model

# In[ ]:


estimator = tf.estimator.DNNClassifier(hidden_units = [32, 32, 32], feature_columns = feature_columns, model_dir = '/estimator', n_classes = 2, activation_fn = 'relu')


# In[ ]:


def input_fn(ds_train):
    return ds_train.unbatch().shuffle(1000).batch(5).repeat()


# In[ ]:


#estimator.train(input_fn = lambda: input_fn(ds_train)) #does not work


# In[ ]:


#So we probably need to create the EagerTensor within the function, so let's try the alternative to define a function that creates the EagerTensor starting from the beginning
def input_fn(numeric_features, train = True, batch_size = 5, num_epochs = 1, shuffle = False, steps = 40):
    ds_raw = tf.data.experimental.make_csv_dataset(filepath, batch_size = batch_size, label_name = 'status', num_epochs = num_epochs, shuffle = shuffle)
    ds = ds_raw.map(lambda features, label: (features, tf.map_fn(lambda x: 1 if x == 'Placed' else 0, label, dtype = tf.int32)))
    
    ds = ds.map(lambda features, label: (popping(features, 'salary'), label))
    ds = ds.map(lambda features, label: (popping(features, 'sl_no'), label)) #drop Serial Number, too
    
    ds = ds.map(PackNumericFeatures(numeric_features))
    
    if train:
        ds = ds.take(steps)
        return ds.unbatch().shuffle(1000).batch(batch_size).repeat()        
    else:
        ds = ds.skip(steps)
        return ds.unbatch().shuffle(1000).batch(batch_size)


# In[ ]:


estimator.train(input_fn = lambda: input_fn(numeric_features, train = True), steps = 400)


# In[ ]:


eval_result = estimator.evaluate(input_fn = lambda: input_fn(numeric_features, train = False))


# In[ ]:


eval_result


# In[ ]:


#The accuracy of 1.0 seems too good to be true...


# # Manually building a Keras Model

# In[ ]:


input_layer = tf.keras.layers.DenseFeatures(feature_columns)


# In[ ]:


model = tf.keras.Sequential()

model.add(input_layer)
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])


# In[ ]:


input_train = ds_train.unbatch().shuffle(1000).batch(5)
model.fit(ds_train, epochs = 20)


# In[ ]:


eval_results2 = model.evaluate(ds_test)


# In[ ]:


#So this time we have a lower test accuracy, we can still predict the result
predictions = model.predict(ds_test)


# In[ ]:


def Sigmoid(x):
    return 1/(1+np.exp(-x))


# In[ ]:


Sigmoid(20)


# In[ ]:


pred = predictions.ravel()
pred_prob = Sigmoid(pred)


# In[ ]:


actual = np.array([])
for batch in ds_test:
    actual = np.append(actual, batch[1].numpy())


# In[ ]:


for p, a in zip(pred_prob, actual):
    print(f'The predicted Placement probability is {p:.2%}, the actual result of the placement is {"Placed" if a == 1 else "Not Placed"}.')


# In[ ]:




