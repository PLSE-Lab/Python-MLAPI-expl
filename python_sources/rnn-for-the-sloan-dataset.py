#!/usr/bin/env python
# coding: utf-8

# # MLP classificer for the sloan data set
# We apply almost _out of the box_ the [mlp classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) packaged with sklearn to the sloan dataset and find it to be very accurate! We give a basic introduction on how to use a multilayer perceptron to classification tasks

# first of all we need to import all the packages we need. Numpy and Pandas for data manipulation and all the modules from sklearn

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

import os
print(os.listdir("../input"))


# We load the dataset from the csv file and we give a peek into what is inside

# In[ ]:


dataset = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')


# In[ ]:


dataset.head()


# Most of the information in the columns are just classification labels. Based on this fact set we can restrict our exploration to columns with physical properties, to be the _redshift_ and the response of the telescope to the electromagnetic bands. Furthermore, we need the class too :)

# In[ ]:


columns = ['redshift', 'u', 'g', 'r', 'i', 'z', 'class']


# The class column contains strings, so we need a label encoder to convert it to numerical values

# In[ ]:


dataset = dataset.loc[:, columns]

le = LabelEncoder().fit(dataset['class'])
dataset['class'] = le.transform(dataset['class'])


# In[ ]:


dataset.head()


# Now we split the dataset intro a training and test. We also perform a simple grid search looking for the better activation function of our network that is, by default, a one layer network with 100 neurons.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataset.drop(labels = 'class', axis = 'columns'), dataset['class'], test_size = 0.3)


# In[ ]:


dici_param = {"activation": ["tanh", "logistic", "relu"]}
clf = GridSearchCV(estimator = MLPClassifier(max_iter=400), param_grid = dici_param, cv = 5, n_jobs = -1)


# In[ ]:


clf.fit(X_train, y_train)


# After training, lets see our score!

# In[ ]:


clf.score(X_test, y_test)


# The neural network can classify correctly more than 90% of the test sample! Let's make a confusion matrix to see where the erros are distributed

# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


class_labels = le.inverse_transform([0,1,2])
confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred),
                            columns = class_labels,
                            index = class_labels)


# In[ ]:


confusion_df

