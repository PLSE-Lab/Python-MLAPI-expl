#!/usr/bin/env python
# coding: utf-8

# This data set includes descriptions of samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. The data files were downloaded from the UCI ML data repository. The data set goes back to 1987.
# 
# Each mushroom species is identified as definitely edible, definitely poisonous, or of unknown edibility. The class of mushrooms with unknown edibility are combined with the poisonous class in the data set. There is no simple rule for determining the edibility of a mushroom.
# 
# The data is in a comma-separated format. There are 8124 observations, with 23 columns. The first column value is 'p' or 'e' (p: poisonous, e: edible), this is the target value. Missing values are flagged with '?'.
# 
# The objective of this analysis is to create a classifier which predicts whether a given sample represents an edible or a poisonous mushroom. We will use this data set to illustrate how to set up the data and build a Neural Network Classifier.
# 
# The 22 attributes are described below.
# Attribute information: (22 attributes)
# 1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 1. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 1. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 1. bruises?: bruises=t,no=f
# 1. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 1. gill-attachment: attached=a,descending=d,free=f,notched=n
# 1. gill-spacing: close=c,crowded=w,distant=d
# 1. gill-size: broad=b,narrow=n
# 1. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
# 1. stalk-shape: enlarging=e,tapering=t
# 1. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
# 1. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 1. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 1. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 1. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 1. veil-type: partial=p,universal=u
# 1. veil-color: brown=n,orange=o,white=w,yellow=y
# 1. ring-number: none=n,one=o,two=t
# 1. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
# 1. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
# 1. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
# 1. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

# We will approach the problem as follows:
# 1. Examine the data. Locate any missing information. Decide how to handle the missing information.
# 1. Prepare the data. Apply one-hot encoding to the categorical variables.
# 1. Build a neural network classifier.

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


# In[ ]:


mushroom_data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')


# In[ ]:


mushroom_data.head()


# Locate all missing values.

# In[ ]:


null_sum = mushroom_data.isnull().sum()


# In[ ]:


null_sum


# There are no remaining null values in the cells. Look for the '?' marker.

# In[ ]:


for name in mushroom_data.columns:
    print(name, mushroom_data[mushroom_data[name] == '?'].shape)


# We find that the 'stalk-root' column has 2480 rows with cells flagged with '?' marker for missing values. We will drop this column from the analysis. The assumption is that we will not lose much information from dropping 1 column out of 22 attributes.

# In[ ]:


mushroom_data.drop(['stalk-root'], axis=1, inplace=True)


# In[ ]:


mushroom_data.shape


# In[ ]:


y = mushroom_data['class']
X = mushroom_data.drop(['class'], axis=1)


# In[ ]:


y.shape


# In[ ]:


X.shape


# Apply one-hot encoding to the categorical variables. This is done by creating one binary attribute per category. It is called one-hot encoding because only 1 attribute will be equal to 1 (hot), while the others will be 0 (cold). The new attributes are sometimes called dummy attributes.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


cat_encoder = OneHotEncoder(sparse=False)
X_1hot = cat_encoder.fit_transform(X)


# In[ ]:


X_1hot[1:5,]


# The target class (p, e) has to be binary encoded.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
y_1hot = lbl_encoder.fit_transform(y)


# In[ ]:


y_1hot


# We will build the neural network using the keras implementation of tensorflow.

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


from sklearn.model_selection import train_test_split


# We will split the data into a training set, a validation set, and a test set.

# In[ ]:


X_train_full, X_test, y_train_full, y_test = train_test_split(X_1hot, y_1hot, test_size=0.33, random_state=1)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


model = keras.models.Sequential()
model.add(Dense(30, activation="relu", input_shape=X_train.shape[1:]))
model.add(Dense(1, activation="sigmoid"))


# In[ ]:


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[ ]:


history = model.fit(X_train, y_train, epochs=20, batch_size=25, validation_data=(X_valid, y_valid))


# In[ ]:


model.evaluate(X_test, y_test)


# We built a classifier with 100% accuracy!
