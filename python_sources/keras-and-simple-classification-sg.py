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


# In[ ]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

## Question 1: Whether to use sci-kit?
## Question 2: Fitting the model
## Question 3: Predicting and inspecting the model.
## Question 4: Loss function for classifers
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)
# * Geron

# In[ ]:


df=pd.read_csv('/kaggle/input/breastcancer/breastcancer.csv')


# In[ ]:


df.head(3)


# In[ ]:


X=df.iloc[:,1:10]
y=df.iloc[:,10]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X_scaled, encoded_y, test_size=0.2, random_state=12)


# In[ ]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(20,input_dim=9, activation="relu"))
model.add(keras.layers.Dense(8, activation="relu"))
#model.add(keras.layers.Dense(25, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))


# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# * If there are 10 classes last layer will be model.add(keras.layers.Dense(10, activation="softmax"))

# In[ ]:


model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])


# In[ ]:


history = model.fit(X_train, y_train, epochs=100)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


# * Binary Crossentropy:  When there are two classes
# * CategoricalCrossentropy : When more than two classes and classes are used as one hot encoding
# * SparseCategoricalCrossentropy: When more than two classes and class labels are provided as integers
# * Some other optimizers are RMSprop, Adam etc.

# In[ ]:


model.summary()


# In[ ]:


model.evaluate(X_test, y_test)

