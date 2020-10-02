#!/usr/bin/env python
# coding: utf-8

# # Data set
# From https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# 3 types of Iris Flowers:
# 
# 1. Iris Setosa
# 2. Iris Versicolour
# 3. Iris Virginica

# # Data columns
# 
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. Species

# # Import packages

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.model_selection import train_test_split

print(tf.__version__)

import os
print(os.listdir("../input"))


# # Data preview

# In[ ]:


iris = pd.read_csv('../input/Iris.csv').set_index('Id')
iris.head(5)


# In[ ]:


iris.info()


# In[ ]:


iris.describe()


# In[ ]:


iris.groupby('Species').count()


# # Data visualization

# In[ ]:


sn.boxplot(x='Species', y='SepalLengthCm', data=iris)


# In[ ]:


sn.boxplot(x='Species', y='SepalWidthCm', data=iris)


# In[ ]:


sn.boxplot(x='Species', y='PetalLengthCm', data=iris)


# In[ ]:


sn.boxplot(x='Species', y='PetalWidthCm', data=iris)


# In[ ]:


sn.pairplot(iris,hue='Species')
plt.show()


# In[ ]:


iris.corr()


# In[ ]:


sn.heatmap(iris.corr(), annot=True)


# # Classification using Tensorflow

# # Split data into train and test

# In[ ]:


X = iris.drop(['Species'],axis=1)
y = iris['Species'].replace({'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2})
print(X.shape, y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(X_train.shape, X_test.shape)


# In[ ]:


# data is shuffled
X_train.head(5)


# In[ ]:


y_train.head(5)


# In[ ]:


# Convert to np arrays so that we can use with TensorFlow
X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.int)
y_test  = np.array(y_test).astype(np.int)

print(X_train)
print(y_train)


# # Feature columns and model

# In[ ]:


feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name,shape=[4])]

classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                          n_classes=3,
                                          model_dir='/tmp/iris_model')


# # Input function

# In[ ]:


def input_fn(X,y):
    def _fn():
        features = {feature_name: tf.constant(X)}
        label = tf.constant(y)
        return features, label
    return _fn

print(input_fn(X_train,y_train)())
        


# # Training

# In[ ]:


classifier.train(input_fn=input_fn(X_train,y_train), steps=1000)
print('fit done')


# # Evaluation

# In[ ]:


accuracy = classifier.evaluate(input_fn=input_fn(X_test,y_test),steps=100)['accuracy']
print('\nAccuracy: {0:f}'.format(accuracy))

