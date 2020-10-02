#!/usr/bin/env python
# coding: utf-8

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


# In this kernel I will use tensorflow to classify iris dataset. In case you are not familiar with the data set read below if not skip it.  
# > The Iris data set contains four features and one label. The four features identify the following botanical characteristics of individual Iris flowers: <br>
# > sepal length <br>
# > sepal width <br>
# > petal length <br>
# > petal width

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


data = pd.read_csv('../input/Iris.csv')


# In[ ]:


# Let's do some data explatory first
data.head()


# In[ ]:


# We don't need Id column
data.drop('Id', axis=1, inplace=True)


# In[ ]:


sns.pairplot(data, hue='Species')


# In[ ]:


# let's turn 'speciaes' into a numeric values becuase this what we need for classification
data["Species"] = data["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})
data.info()


# ## Tensorflow

# Algorithm:
# The program trains a Deep Neural Network classifier model having the following topology:<br>
# * 2 hidden layers.
# * Each hidden layer contains 10 nodes.
# ![](https://www.tensorflow.org/images/custom_estimators/full_network.png)
# 
# Running the trained model on an unlabeled example yields three predictions, namely, the likelihood that this flower is the given Iris species. The sum of those output predictions will be 1.0. For example, the prediction on an unlabeled example might be something like the following:
# * 0.03 for Iris Setosa
# * 0.95 for Iris Versicolor
# * 0.02 for Iris Virginica <br>
# The preceding prediction indicates a 95% probability that the given unlabeled example is an Iris Versicolor.

# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[ ]:


y = data['Species']
X= data.drop('Species', axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# First thing we need is an estimator. You may either use the pre-made Estimators tensorflow provides or write your own  Estimators. To write a TensorFlow program based on pre-made Estimators, we must perform the following tasks:
# 
# * Define the model's feature columns.
# * Create an input function.
# * Instantiate the Estimator<br> 
# 
# Let's see how those tasks are implemented for Iris classification.

# # Feature columns
# A feature column describes each of the features you want the model to use. For Iris, there are  4 raw features (sepal length, sepal width, petal length, petal width) and they are numeric values, so we are goingto select numeric columns. <br>
# The tf.feature_column module provides many options for representing data to the model.

# In[ ]:


feat_cols = [tf.feature_column.numeric_column(col) for col in X.columns]


# In[ ]:


feat_cols[:3]


# # Input Function
# The next step is to create input functions to supply data for training, evaluating, and prediction. An input function is a function that returns a tf.data.Dataset object which outputs the following two-element tuple:
# * features - A Python dictionary in which:<br>
# Each key is the name of a feature.<br>
# Each value is an array containing all of that feature's values.<br>
# * label - An array containing the values of the label for every example.<br>
# 
# There are two ways to create input functions. first use ***tf.estimator.inputs.***  you can either pick pandas_input_fn or numpy_input_fn <br>
# Second build your own input function. Input functions must return a mapping of the feature columns to the tensors that contains the data for the feature columns along with labels.

# In[ ]:


# First way using tf.estimator.inputs. NOTE: you have to provide shuffle argument 
# otherwise you will get an error. Since we used trains test split I already shuffled it
input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, 
                                                 num_epochs=5, shuffle=False)

# # second way: build your own function
# def input_fn(df,labels):
#     feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in X.columns}
#     label = tf.constant(labels.values, shape = [labels.size,1])
#     return feature_cols,label


# # Instantiate an estimator
# Fortunately, TensorFlow provides several pre-made classifier Estimators. We use Deep Neural Network Classifier (DNNClassifier) for iris problem.  <br> 
# DNNClassifier is instantiated by providing feature_columns,  number of hidden units and number of classes as parameters. 

# In[ ]:


#We added 3 hidden layer with 10 20 and 10 nodes respectively. 
# We are overkilling for such a simple problem. You can test different hiddent unites and nodes 
classifier = tf.estimator.DNNClassifier(hidden_units=[10,10], n_classes=3, 
                                        feature_columns=feat_cols)


# # Train, Evaluate, and Predict
# Now that we have an Estimator object, we can do followings:
# 
# * Train the model.
# * Evaluate the trained model.
# * Make predictions using the trained model .

# In[ ]:


# Train the estimator run the following if you used tf.estimator.inputs
classifier.train(input_fn=input_fn, steps=50)

# # if you build your own input funtion above, you will need to run the following
# classifier.train(input_fn=lambda:input_fn(X_train, y_train), steps=50)


# In[ ]:


# In order to evaluate model I need to  creatr another input function for evaluation
# I will use tf.estimator.inputs.pandas_input_fn
eval_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=len(X_test), 
                                              shuffle=False)

# # if you want to use input_fn you built, you will need to run the following
# # this time pass X_test, y_test to you input function
# classifier.evaluate(input_fn=lambda:input_fn(X_test, y_test), steps=50)


# # Evaluate the trained model
# Now that the model has been trained, we can get some statistics on its performance. Unlike train method, we don't need to pass the steps argument to evaluate. Our eval_fn only yields a single epoch of data.

# In[ ]:


eval_result = classifier.evaluate(input_fn=eval_fn)
print(eval_result)


# # Making predictions
# We can now use the trained model to predict the species of an Iris flower based on some unlabeled measurements. 

# In[ ]:


# we will create another input function for predictions and call it pred_fn
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)


# In[ ]:


# classifier.predict is a generator, we need cast result to a list
predictions = list(classifier.predict(input_fn=pred_fn))


# In[ ]:


predictions[:3]
# value for the 'class ids' is the selected class


# In[ ]:


# let's collect them in a list
final_pred = [pred['class_ids'][0] for pred in predictions]


# In[ ]:


final_pred[:10]


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(confusion_matrix(y_test, final_pred))


# In[ ]:


print(classification_report(y_test, final_pred))


# In[ ]:


# it seems it is doing pretty good


# In[ ]:


# question can we do as good as NN with a simpler algorithms such as decision tree or 
# logistic regression. i will answer this in the second part but for now that is it


# In[ ]:




