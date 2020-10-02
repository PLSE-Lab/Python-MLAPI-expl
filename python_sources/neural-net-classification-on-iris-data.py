#!/usr/bin/env python
# coding: utf-8

# # Neural Network Classification on Iris Data
# [Previously](https://github.com/bbevan/Colabs/blob/master/K_Means_Speciation.ipynb), I ran a K-Means classification algorithm on this data, which achieved a success rate of ~88%. I'd like to know if a simple Neural Network can do better.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Requirements
# Instead of using the Tensorflow library directly, I'm using the Keras API to Tensorflow's routines.

# In[ ]:


# Requirements
import keras
from keras import Sequential

# Turn off complaints
import warnings
warnings.filterwarnings("ignore")


# # Data
# The same data is used from the K-Means experiment. In short, it is the Iris flower data collected by R.A. Fisher to determine Taxanomic classification by Sepal and Petal measurements. The model will use four columns of measurements (sepal and petal lengths and widths)  as factors for predicting species labels.

# In[ ]:


# Import Data
df = pd.read_csv('../input/Iris.csv')


# `head` is a nice function for making sure the data is sane.

# In[ ]:


# Check it out
df.head()


# # Viewing pairplots
# Using `Seaborn` , a visualization library for Python, we can see scatterplots of each variable against the other.

# In[ ]:


# EDA
import seaborn as sns

sns.pairplot(df, hue="Species")


# As a reminder:
# * Sepal lengths and Sepal width are entangled within each other.
# * K-Means was able to predict the correct species with 88% accuracy
# * The goal in this experiment is to find out if a Neural Network can identify the correct species with greater accuracy despite the same entangled data.

# # Data preparation
# Since this is a supervised learning task, we need to distinguish between the targets and the predictive factors.
# * `targets` - contains the Species labels.
# * `df` - will contain the Sepal and Petal measurements by dropping `Species` and `Id` in place.
# 
# The data is then normalized by the `sklearn` `preprocessing` routines.

# In[ ]:


# Data Prep

# Manip
targets = df["Species"]
df.drop(["Species", "Id"], axis = 1, inplace=True)


# Normalize
from sklearn import preprocessing

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)


# # One-Hot Encoding
# The target values are categorical in nature. 
# * Neural Networks are capable of working with categorical variables.
# * They must be encoded numerically, however.
# * One-hot encoding encodes categorical variables as binary arrays.

# In[ ]:


# Convert Targets to One-Hots
targets = targets.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                [0,1,2])

targets = keras.utils.np_utils.to_categorical(targets, num_classes=3)


# Lets just make sure everything is ok at this point with `len`.

# In[ ]:


# Sanity
len(df)


# # Neural Network Model
# The NN is sequential, built with Dense layers.
# * The input layer contains 10 nodes, with an input shape of 4 corresponding to the four predictive variables.
# * There is one hidden layer, 10 nodes, with non-linearities introduced by the `relu` function.
# * The output layer has 3 nodes. We use the `sigmoid` function as opposed to `softmax` here.
# 
# The model is compiled with the following settings:
# * `optimizer` - `rmsprop` as opposed to `adam`.
# * `loss` - `categorical_crossentropy` since this is a categorical prediction problem.
# * `metrics` - `acc` for a 0.0 -1.00 accuracy score.
# 
# The model is then trained with the following settings:
# * x values are chosen from `df`
# * y values are the `targets`
# * 100 epochs
# * `batch_size` of 2 
# * `validation_split` on 10 percent of the data

# In[ ]:


# Build Neural Network
from keras.layers import Dense

# Create a new Sequential object
model = Sequential()

# Create the input layer, 50 nodes
model.add(Dense(10, input_shape=(4,)))

# Create the hidden layer
model.add(Dense(10, activation="relu"))
          
# Create an output layer, 3 nodes
model.add(Dense(3, activation="sigmoid"))

# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# Train the NN model
model.fit(x=df, y=targets, epochs=100, batch_size=2, validation_split=0.10)


# # Model evaluation
# I've evaluated the model much like the evaluation of the K-Means model. I'm just re-predicting the species labels based on the entire dataset. In reality, this is never done due the fact that the accuracy score would be artificially inflated due to overfitting. But I'm hoping that this method returns a value comparable to the K-Means acuracy measure.

# In[ ]:


# Evaluate model
model.evaluate(x=df, y=targets, batch_size=16)


# About 95% Accurate.
# 
# *Caveat: the code may return different accuracy readings when re-run.*

# # Predicting the class labels
# The Keras API has a built in method for predicting class labels
# * These class labels are the Species identifiers.

# In[ ]:


preds = model.predict_classes(df)

for i in range(len(df)):
    print("Prediction = %s, Actual = %s" % (preds[i], targets[i]))


# # Visualizations

# In[ ]:


# Manip
data = df.copy()
data["preds"] = preds


# In[ ]:


sns.pairplot(data, hue="preds", vars=[0,1,2,3])


# As you can see from the Pair Plot above, this particular model recognized all three species. In the original Neural Network code that I built, only two species were identified due to the entanglements metioned above.

# Shoutout to [this notebook](https://www.kaggle.com/nityansuman/iris-deep-classifier) , which helped me diagnose a few issues.

# 
