#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('../input/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbl_encoder_Country = LabelEncoder()
X[:, 1] =lbl_encoder_Country.fit_transform(X[:, 1])
lbl_encoder_gender = LabelEncoder()
X[:, 2] = lbl_encoder_gender.fit_transform(X[:, 2])
onehotencoder_country = OneHotEncoder(categorical_features=[1])
X = onehotencoder_country.fit_transform(X).toarray()
# dropping Ist dummy variable of country to avoid dummy variable trap
X = X[:, 1:]


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(copy=True, with_mean=True, with_std=True) # calcuating the Z score
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


# Improving the Accuracy using k-fold Cross Validation
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


def build_classifier(units, optimizer):
    classifier = Sequential()
    classifier.add(Dense(units= units, kernel_initializer='uniform', activation='relu' , input_dim=11))
#     classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units= units, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units= units, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# In[ ]:


# classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
# print(accuracies)


# In[ ]:


# taking the mean of the accuracies
# mean = accuracies.mean()
# print(mean)
# calculating the variance
# variance = accuracies.var()
# print(variance)


# In[ ]:


# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'units': [6, 12, 18],
               'batch_size' : [35, 42],
              'epochs' : [100, 500],
              'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                          param_grid= parameters,
                          scoring = 'accuracy',
                          cv = 10 # cross-validation
                          )
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print('Best Parameters : ', best_parameters)
best_accuracy = grid_search.best_score_
print('Best accuracy : ', best_accuracy)


# In[ ]:




