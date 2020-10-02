#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U tf-nightly')
import tensorflow as tf
tf.__version__


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pip show tensorflow keras


# Example from Structured data classification @keras.io/examples
# 
# Author: Francois Chollet
# 
# On: 9th June, 2020

# # Data description:
# 
# **Age** Age in years
# 
# **Sex** (1 = male; 0 = female)
# 
# **CP** Killip's Chest pain class (0, 1, 2, 3, 4)
# 
# **Trestbpd** Resting BP Systolic (mm Hg on admission)
# 
# **Chol** S. Cholesterol (mg / dl)
# 
# **FBS** Fasting bl. sugar (>120 mg/dl) (1 = true; 0 = false)
# 
# **RestECG** Resting ECG (0, 1, 2)
# 
# **Thalach** Max HR on exercise
# 
# **Exang** Exercise induced angina (1 = yes; 0 = no)
# 
# **Oldpeak** ST depression?? induced by exercise (range: 0 - 6.2)
# 
# **Slope** Slope ST segment by exercise (1 = normal; 2 = abnormal)
# 
# **CA** No. of major vessels (0 - 3) affected on fluoroscopy
# 
# **Thal** 1 = normal; 2 = fixed defect (not 6); 3 = reversible defect (not 7)
# 
# **Target** Ischaemic Heart Disease (1 = yes; 0 = no)

# # Load data

# In[ ]:


## load data
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
print(df.shape)
df.sample(10)


# # Pre-processing

# In[ ]:


# test-train split
val_df = df.sample(frac = 0.1, random_state = 42)
train_df = df.drop(val_df.index)

print(
      "Using %d samples for training and %d for validation"
      % (len(train_df), len(val_df))
)


# In[ ]:


# generate tf.data.Dataset objects for each dataframe
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size = len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_df)
val_ds = dataframe_to_dataset(val_df)

# Each Dataset yields a tuple (input, target) where input is a dictionary of features and target is the value 0 or 1
for x, y in train_ds.take(2):
    print("Input:", x)
    print("Target:", y)

# only oldpeak is of float64 type, else all other vars are of int64 type


# In[ ]:


# batch the datasets
train_ds = train_ds.batch(8)
val_ds = val_ds.batch(8)


# In[ ]:


# encoding functions
def encode_numerical_feature(feature, name, dataset):
    # Create a Normalisation layer for our feature
    normaliser = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normaliser.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normaliser(feature)
    return encoded_feature

def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode = "binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature

def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode = "binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
    return encoded_feature


# In[ ]:


# Categorical features encoded as integers
sex = keras.Input(shape =(1, ), name = "sex", dtype = "int64")
fbs = keras.Input(shape =(1, ), name = "fbs", dtype = "int64")
restecg = keras.Input(shape =(1, ), name = "restecg", dtype = "int64")
exang = keras.Input(shape =(1, ), name = "exang", dtype = "int64")
ca = keras.Input(shape =(1, ), name = "ca", dtype = "int64")
thal = keras.Input(shape =(1, ), name = "thal", dtype = "int64")
slope = keras.Input(shape =(1, ), name = "slope", dtype = "int64")

# Numerical features
oldpeak = keras.Input(shape =(1, ), name = "oldpeak", dtype = "float64")
age = keras.Input(shape =(1, ), name = "age", dtype = "int64")
cp = keras.Input(shape =(1, ), name = "cp", dtype = "int64")
trestbps = keras.Input(shape =(1, ), name = "trestbps", dtype = "int64")
chol = keras.Input(shape =(1, ), name = "chol", dtype = "int64")
thalach = keras.Input(shape =(1, ), name = "thalach", dtype = "int64")

all_inputs = [
    sex,
    cp,
    fbs,
    restecg,
    exang,
    ca,
    thal,
    age,
    trestbps,
    chol,
    thalach,
    oldpeak,
    slope,
]

# Integer categorical features
sex_encoded = encode_integer_categorical_feature(sex, "sex", train_ds)
fbs_encoded = encode_integer_categorical_feature(fbs, "fbs", train_ds)
restecg_encoded = encode_integer_categorical_feature(restecg, "restecg", train_ds)
exang_encoded = encode_integer_categorical_feature(exang, "exang", train_ds)
ca_encoded = encode_integer_categorical_feature(ca, "ca", train_ds)
thal_encoded = encode_integer_categorical_feature(thal, "thal", train_ds)
slope_encoded = encode_integer_categorical_feature(slope, "slope", train_ds)

# Numerical features (mean = 0, sd = 1)
oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
age_encoded = encode_integer_categorical_feature(age, "age", train_ds)
cp_encoded = encode_integer_categorical_feature(cp, "cp", train_ds)
trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)

all_features = layers.concatenate(
    [
        sex_encoded,
        cp_encoded,
        fbs_encoded,
        restecg_encoded,
        exang_encoded,
        slope_encoded,
        ca_encoded,
        thal_encoded,
        age_encoded,
        trestbps_encoded,
        chol_encoded,
        thalach_encoded,
        oldpeak_encoded,
    ]
)

x = layers.Dense(8, activation = "relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation = "sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# # Plot & fit model

# In[ ]:


keras.utils.plot_model(model, show_shapes = True, rankdir = "LR")


# In[ ]:


model.fit(train_ds, epochs = 64, validation_data = val_ds)


# We obtain a validation accuracy of 90%

# # Prediction on a new example

# In[ ]:


sample = {
    "age": 50,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 120,
    "fbs": 0,
    "restecg": 0,
    "thalach": 160,
    "exang": 0,
    "oldpeak": 6,
    "slope": 1,
    "ca": 0,
    "thal": 1,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
model.predict(input_dict)


# Based on the model, the person has 1% chance of having Ischaemic Heart Disease.
