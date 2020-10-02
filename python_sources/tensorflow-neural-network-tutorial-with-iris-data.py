#!/usr/bin/env python
# coding: utf-8

# Tensorflow's high level API(tf.contrib.learn) makes it easy to create, fit and evaluate many out-of-the-box models. It includes linear classifier/regressor, fully connected neural networks , combined deep and wide models etc. Here we'll work with iris dataset to create a DNNClassifier and evaluate it based on petal length/width and sepal length/width. 
# 
# Steps would be : 
# 
# * Converting the categorical labels to integers
# * Creating training/test set
# * Converting the features to tensors
# * Creating the input function to feed the features into the model
# * Constructing the [neural network classifier][1]
# *  Fitting and Evaluating the Classifier
# * Generating Predictions
# 
#   [1]: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.head()


# In[ ]:


iris.shape


# In[ ]:


iris.dtypes


# In[ ]:


iris.iloc[:,1:4] = iris.iloc[:,1:4].astype(np.float32)


# In[ ]:


iris.dtypes


# # Convert categorical label to  integers 

# In[ ]:


iris["Species"] = iris["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})


# # Split the data into training and test set 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:,1:5], iris["Species"], test_size=0.33, random_state=42)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# # Construct, fit and evaluate the classifier
# 
# ``` DNNClassifier``` expects following arguments : 
# 
# * ```feature_columns``` : Feature columns map the data to the model. We can either use raw features from the training dataset or any derived features from them. See [here](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.layers/feature_columns) for more information. 
# 
# * ``` hidden_units``` : List containing number of hidden units in each layer. All layers would be fully connected. 
# 
# * ``` n_classes ``` : Number of classes 
# 
# Optionally we can also set the optimizer, dropout and activation functions. Default activation function is ReLu. If we set a model directory then it'd save the model graph, parameters etc. See the documentation for reading up on [DNNClassifier](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.layers/feature_columns)

# In[ ]:


columns = iris.columns[1:5]


# In[ ]:


print(columns)


# In[ ]:


import tensorflow as tf


# # Create the Feature Columns

# All of the features in our training dataset are real valued and continuous. Thus we can use ```tf.contrib.layers.real_valued_columns``` to construct the feature columns here. We can use [Sparse Tensors](https://www.tensorflow.org/api_guides/python/sparse_ops) for categorical features. 

# In[ ]:


feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]


# # Make input function
# 
# Estimators in tf.contrib.learn can accept input functions that feed the data to the model during training. Input functions must return a mapping of the feature columns to the tensors that contains the data for the feature columns along with labels. Here we create a dictionary where the keys are the feature columns which map to the tensors containing values for these features. Note that we've to return both feature columns and labels. 
# 
# In this case we can use just one ```input_fn``` and pass the dataframe into it instead of creating seperate functions for the training and the testing set. 

# In[ ]:


def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label


# # Construct the Classifer 
# 
# Here the ```DNNClassifier``` is constructed with the feature_columns where the number of hidden units in each layer is 10,20,10 respectively. The number of classes is also specified.

# In[ ]:


classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes = 3)


# # Fit the classifier 
# We pass the input_fn into the ```fit``` method and set the number of steps.

# In[ ]:


classifier.fit(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)


# # Evaluate the Classifier 
# 
# ```Evaluate``` method returns some statistics like accuracy, auc after being called on the test data

# In[ ]:


ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)


# In[ ]:


print(ev)


# # Generate predictions :
# 

# In[ ]:


def input_predict(df):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    return feature_cols


# In[ ]:


pred = classifier.predict_classes(input_fn=lambda: input_predict(X_test))


# In[ ]:


pred


# ```classifier.predict_classes``` return a generator object but we can convert to a list for getting the predictions for now.

# In[ ]:


print(list(pred))


# References  : https://www.tensorflow.org/get_started/tflearn

# In[ ]:




