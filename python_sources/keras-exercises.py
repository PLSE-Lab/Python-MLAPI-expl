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

#import dataset
dataset = pd.read_csv('../input/creditcard.csv')
X = dataset.iloc[:, 1:29].values  # leave timedate (1st) column out
y = dataset.iloc[:, 30].values

#split dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Compile and fit ANN

# import keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#########################
# extract auc from tf and use it as a measure in keras (B. Kanani's sample code)
import tensorflow as tf
from keras import backend as K
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
#########################

# initialize ANN and add layers
classifier = Sequential()
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 28))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compile classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc',auc])

#fitting the ANN classifier to training set
history = classifier.fit(X_train, y_train, shuffle = True, validation_split = 0.2,  batch_size = 100, epochs = 10)


#test set performance
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#output confusion matrix to csv
pd.DataFrame(cm).to_csv('confusion_matrix_results_auc.csv')

