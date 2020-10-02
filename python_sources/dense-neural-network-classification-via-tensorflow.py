# -*- coding: utf-8 -*-
"""
Created on Tue May 01 22:38:42 2018

@author: Muhammad Salek Ali
"""

# Dense Neural Network Classification on Tensorflow

# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import os
print(os.listdir("../input"))

# Import data and normalize columns
data = pd.read_csv("../input/adult_full.csv")

cols_to_norm = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max()-x.min()) )

# Handle the columns for TF 
feat_cols=[]
for i in range(len(data.columns)-1):
    feat_cols.append(tf.feature_column.numeric_column(data.columns[i]))
    
# Train-test split
input_x = data.drop('incoming_classification',axis=1)
input_y = data['incoming_classification']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size = 0.25, random_state = 0)


# write an input function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Dense neural network means every neuron is connected to every neuron in the next stage
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10], feature_columns=feat_cols, n_classes=2)
# no need to create embedded columns, all columns are already embedded

# PULL THE LEVER, KRONK!
dnn_model.train(input_fn=input_func, steps=1000)

# Evaluate the model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = dnn_model.evaluate(eval_input_func)
print(results)

# Make some predictions
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = dnn_model.predict(pred_input_func)

my_pred=list(predictions)
