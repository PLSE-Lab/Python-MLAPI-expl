#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.feature_selection import RFE
from IPython.core.interactiveshell import InteractiveShell
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()
test.head()
train.shape
test.shape


# In[ ]:


train.PassengerId.value_counts()
train.Sex.value_counts()
train.Pclass.value_counts()
train.Name.value_counts()
train.Embarked.value_counts()
train.Cabin.value_counts()
train.SibSp.value_counts()
train.Ticket.value_counts()
train.Parch.value_counts()
train.Fare.value_counts()
train.Age.value_counts()


# # Name and PassengerId are removed since they both are unique and ticket is removed since it has no affect on survived

# In[ ]:


train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Name'], axis=1)
train = train.drop(['Ticket'], axis=1)
train.head()


# In[ ]:


train['Sex'] = train['Sex'].apply(lambda x: 0 if x == 'male' else 1)
for i in train['Pclass']:
    if i == 1:
        train['Pclass'] = train['Pclass'].replace(i, 'Upper')
        
    elif i == 2:
        train['Pclass'] = train['Pclass'].replace(i, 'Middle')

    else:
        train['Pclass'] = train['Pclass'].replace(i, 'Lower')
        
train.head()


# In[ ]:


train.isnull().any()


# In[ ]:


train.Cabin.isnull().sum()
train.Embarked.isnull().sum()
train.Age.isnull().sum()


# # Too many missing values for Cabin. This feature is not useful so performing column-wise deletion

# In[ ]:


train = train.drop(['Cabin'], axis=1)
train.head()


# # Drop duplicates

# In[ ]:


train = train.drop_duplicates()
train


# # Row-wise deletion for missing Embarked feature

# In[ ]:


train = train.dropna(subset=['Embarked', 'Age'])
train.Age.fillna(train.Age.mean())
train.isnull().any()


# In[ ]:


train.head()
train.shape


# In[ ]:


plt.scatter(x = train['Age'], y=train['Age'])
plt.show()
plt.scatter(x = train['Fare'], y=train['Fare'])
plt.show()
plt.scatter(x = train['SibSp'], y=train['SibSp'])
plt.show()
plt.scatter(x = train['Parch'], y=train['Parch'])
plt.show()


# # Remove outliers

# In[ ]:


train = train[train.Fare < 300]
train.shape


# # One hot encode variables

# In[ ]:


train = pd.get_dummies(train, columns=['Pclass', 'Embarked', 'SibSp', 'Parch'])
train.head()


# In[ ]:


train.Age = train.Age.astype(int)
train.Fare = train.Fare.astype(int)
train.head()


# # Drop features that are linearly dependent of each other

# In[ ]:


train = train.drop(['Pclass_Lower'], axis=1)
train = train.drop(['Embarked_C'], axis=1)

train.head()


# In[ ]:


train.SibSp_1
train.SibSp_2


# In[ ]:


train = train.drop(['SibSp_0'], axis=1)
train.head(20)


# In[ ]:


train = train.drop_duplicates()
train.shape


# In[ ]:


train.head(20)


# # Build models

# In[ ]:


sns.countplot(train.Survived)
train.Survived.value_counts()


# In[ ]:


features = ['Age', 'Embarked_Q', 'Embarked_S', 'Fare', 'Parch_0', 'Pclass_Middle', 'Pclass_Upper', 'Sex', 'SibSp_1', 
            'SibSp_2', 'Parch_1', 'Parch_2', 'SibSp_4', 'SibSp_3', 'SibSp_5', 'Parch_4', 'Parch_3', 'Parch_5', 'Parch_6', 'Survived']
train = train[features]
X = train.drop(['Survived'], axis=1)
Y = train['Survived']


# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=0)


# # Random Forest Model

# In[ ]:


rf = RandomForestClassifier()
rf.fit(train_features, train_labels)
rf_pred_train = rf.predict(train_features)
rf_pred_test = rf.predict(test_features)
print(classification_report(test_labels,rf_pred_test))
print('Random Forest baseline: ' + str(roc_auc_score(train_labels, rf_pred_train)))
print('Random Forest: ' + str(roc_auc_score(test_labels, rf_pred_test)))


# # Hyper parameter tuning and feature selection

# In[ ]:


rf = RandomForestClassifier()
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [10],
    'min_samples_leaf': [3],
    'n_estimators': [1000],
    'oob_score': [True],
    'random_state': [0],
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                      cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

# Fit the grid search to the data
grid_search.fit(train_features, train_labels);

grid_search.best_params_
best_grid = grid_search.best_estimator_
pprint(best_grid.get_params())

selector = RFE(rf, step=1, verbose=3)
selector = selector.fit(train_features, train_labels)
print("Features sorted by their rank:")
pprint(sorted(zip(map(lambda x: round(x, 4), selector.ranking_), X)))


# In[ ]:


rf = RandomForestClassifier(**best_grid.get_params())
rf.fit(train_features, train_labels)
rf_pred_train = rf.predict(train_features)
rf_pred_test = rf.predict(test_features)
print(classification_report(test_labels,rf_pred_test))
print('Random Forest baseline: ' + str(roc_auc_score(train_labels, rf_pred_train)))
print('Random Forest: ' + str(roc_auc_score(test_labels, rf_pred_test)))


# # Neural Network

# In[ ]:


model = Sequential()
model.add(Dense(12, input_dim=19, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_features, train_labels)
model_pred_train = model.predict(train_features)
model_pred_test = model.predict(test_features)
print(classification_report(test_labels,rf_pred_test))
print('Neural Network baseline: ' + str(roc_auc_score(train_labels, model_pred_train)))
print('Neural Network: ' + str(roc_auc_score(test_labels, model_pred_test)))


# In[ ]:


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(1, input_dim=19, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(2)))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    optimizer = optimizers.adam(lr=.001)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# create model
model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=10)
# define the grid search parameters
param_grid = {
#     'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#     'batch_size': [10, 20, 40, 60, 80, 100],
#     'epochs': [10, 50, 100],
#     'learn_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
#     'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
#     'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
#     'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
#     'weight_constraint': [1, 2, 3, 4, 5],
#     'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     'neurons': [1, 5, 10, 15, 20, 25, 30]
    
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                      cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

# Fit the grid search to the data
grid_result = grid_search.fit(train_features, train_labels);

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


model.fit(train_features, train_labels)
model_pred_train = model.predict(train_features)
model_pred_test = model.predict(test_features)
print(classification_report(test_labels,rf_pred_test))
print('Neural Network baseline: ' + str(roc_auc_score(train_labels, model_pred_train)))
print('Neural Network: ' + str(roc_auc_score(test_labels, model_pred_test)))

