# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import Normalizer,LabelBinarizer
from sklearn.neural_network import MLPClassifier

import os
print(os.listdir("../input"))

# Loding training and testing Data
training_data = pd.read_csv("../input/training.csv")
testing_data = pd.read_csv("../input/testing.csv")


y_train = training_data['class']
y_test = testing_data['class']

x_train = training_data[['b1','b2','b3','b4','b5','b6','b7','b8','b9']]
x_test = testing_data[['b1','b2','b3','b4','b5','b6','b7','b8','b9']]

# converting the class into binary format
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

# Transforming the data into Normalize
transformer = Normalizer().fit(x_train)
x_train = transformer.transform(x_train)
x_test = transformer.transform(x_test)

# MLP classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(9,6), random_state=1,learning_rate='adaptive')
clf.fit(x_train,y_train)

# predicting on test data
y_pred = clf.predict(x_test)
# print(y_pred)

# converting the binary to class
y_pred = lb.inverse_transform(y_pred)
# print(y_pred)

# classifier report
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
