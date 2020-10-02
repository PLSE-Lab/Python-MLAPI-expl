#!/usr/bin/env python
# coding: utf-8

# # Tensorflow instagram fake accounts detector
# 
# For this classification problem, I decided to use Tensorflow and built a neural network with 3 dense layers (1 input, 1 hidden, 1 output). The result is a model with an accuracy of ~80%

# ## Data
# 
# The dataset contains accounts features such as: # of followers, # of following, presence of profile picture etc.
# The label 'fake' is a value 0 (real profile) or 1 (fake profile).
# 
# The data is available for download here: `kaggle datasets download -d free4ever1/instagram-fake-spammer-genuine-accounts`.

# ## Data pre processing and overview

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from numpy.random import seed
import tensorflow as tf


# In[ ]:


dftrain = pd.read_csv('../input/instagram-fake-spammer-genuine-accounts/train.csv')
dftest = pd.read_csv('../input/instagram-fake-spammer-genuine-accounts/test.csv')

df = pd.concat([dftrain, dftest], axis=0, sort=True)
df.head()


# In[ ]:


sns.countplot(x='profile pic', data=df, palette='hls', hue='fake')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.countplot(x='private', data=df, palette='hls', hue='fake')
plt.xticks(rotation=45)
plt.show()


# There are 11 features in total and 1 categorical label. 
# Some of them are categorical and some of them continuous, so we can scale the continous feature to prevent them from messing up with the prediction.
# 

# In[ ]:


# Scale Continuous Features
continuous_features = ['nums/length username', 'description length', '#posts', '#followers', '#follows']

scaler = StandardScaler()
for feature in continuous_features:
    df[feature] = df[feature].astype('float64')
    df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))

dftrain.head()


# In[ ]:


# Let's create our train test split
X_train = df[pd.notnull(df['fake'])].drop(['fake'], axis=1)
y_train = df[pd.notnull(df['fake'])]['fake']
X_test = df[pd.isnull(df['fake'])].drop(['fake'], axis=1)


# ## Neural Network Model
# 
# The model consists of 3 layers. The input layer contains 11 perceptrons, one for each feature of the dataset.
# The hidden layer is densely connected and has 22 neurons.
# Finally the output layer only contains 1 output neuron for the final prediction.

# In[ ]:


model = Sequential()
model.add(Dense(11, input_dim=X_train.shape[1], activation='linear', name='input_layer'))
model.add(Dense(22, activation='linear', name='hidden_layer'))
model.add(Dropout(0.0))
model.add(Dense(1, activation='sigmoid', name='output_layer'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# ## Results

# The model has been tested at several epochs number, and it seems to peek its performance at ~10 epochs with an accuracy oscillating around 80%.
# Some factors that can increase the overall accuracy are random seeding, which also helps with reproducibility. 

# In[ ]:


training = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['accuracy'])
print("\n%s: %.2f%%" % ('accuracy', val_acc*100))


# In[ ]:


# summarize history for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:




