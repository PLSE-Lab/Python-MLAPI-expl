#!/usr/bin/env python
# coding: utf-8

# ## Classification problem.
# Since this problem is about finding whether a person will live or die in ship wrek. Basically it is a binary clasification (0 or 1, yes or no).
# We have set of algorithms which does that, we will start from those algorithms.
# #### Available Algorithms
# * Logistic regression
# * Naive Bayes
# * Stochastic Gradient Descent
# * KNN
# * SVM
# * Decission tress
# * Random Forest
# * Nueral Nets
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


TRAIN_PATH = '../input/titanic/train.csv'
TEST_PATH = '../input/titanic/test.csv'
LABELS_COL = 'Survived'


# In[ ]:


train = pd.read_csv(TRAIN_PATH)
train.head(5)


# In[ ]:


train.info()


# In[ ]:


train['Cabin'].value_counts()
train['Cabin'].isna().sum() # Create a category of for unkown cabin.


# In[ ]:


train['Age'].isna().sum() # Fill by average age.


# In[ ]:


train['Embarked'].value_counts()
train['Embarked'].isna().sum() # Fill by most frequent.


# In[ ]:


train.describe()


# In[ ]:


labels = train[LABELS_COL]
print('Length of training set is {}'.format(len(labels)))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)


# ## Data preprocessing pipelines using custom transformers in scikit learn
# * Remove unneccesary columns
# * Fill unknowns
# * Convert strings to numbers
# * Normalize the data

# In[ ]:


class SqueezeDataset(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.dummy_cols = columns
    
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        X = X.drop(self.dummy_cols, axis=1)
        return X
    
    
class FillNas(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_col = 'Age'
        self.cabin_col = 'Cabin'
        self.embarked_col = 'Embarked'
    
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        age_mean = X[self.age_col].mean()
        X[self.age_col].fillna(age_mean, inplace=True)
        
        unique_cabins = list(set(X[self.cabin_col]))
        X[self.cabin_col] = [unique_cabins.index(c) for c in X[self.cabin_col]]
        
        most_frequent_embarking = X[self.embarked_col].value_counts()[0]
        X[self.embarked_col].fillna(most_frequent_embarking, inplace=True)
        
        unique_embarking = list(set(X[self.embarked_col]))
        X[self.embarked_col] = [unique_embarking.index(e) for e in X[self.embarked_col]]
        
        return X
    
class Binarize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gender_col = 'Sex'
        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):
        X[self.gender_col] = [1 if s == 'male' else 0 for s in X[self.gender_col]]
        return X


# In[ ]:


data_pipline = Pipeline([
    ('squeeze', SqueezeDataset([LABELS_COL, 'Name', 'Ticket', 'Fare', 'PassengerId'])),
    ('fillnas', FillNas()),
    ('binarize', Binarize()),
    ('scale', StandardScaler())
])


# In[ ]:


preprocesed_training_set = data_pipline.fit_transform(X_train)
preprocesed_training_set


# In[ ]:


preprocesed_training_set.shape


# In[ ]:


logistic_classifier = LogisticRegression(random_state=0, solver='lbfgs')
naive_b_classifer = GaussianNB()
svm_classifier = SVC(kernel='linear', random_state=0)
sgd_ckassifier = SGDClassifier()
knn_classifier = KNeighborsClassifier()
decision_tree_classifier = DecisionTreeClassifier()
random_forest_classifier = RandomForestClassifier(n_estimators=200)


# In[ ]:


models = [logistic_classifier, naive_b_classifer, svm_classifier, sgd_ckassifier, knn_classifier, decision_tree_classifier, random_forest_classifier]
confusion_matrices = []
accuracies = []
test_set = data_pipline.fit_transform(X_test)
for i, m in enumerate(models):
    m.fit(preprocesed_training_set, y_train)
    y_pred = m.predict(test_set)
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)
    ac = accuracy_score(y_test, y_pred)
    accuracies.append(ac)


# In[ ]:


accuracies


# ## Preprocess test data set

# In[ ]:


test = pd.read_csv(TEST_PATH)


# In[ ]:


test.info()


# In[ ]:


test_pipeline = Pipeline([
    ('squeeze', SqueezeDataset(['Name', 'Ticket', 'Fare', 'PassengerId'])),
    ('fillnas', FillNas()),
    ('binarize', Binarize()),
    ('scale', StandardScaler())
])
preproced_test_data = test_pipeline.fit_transform(test)


# In[ ]:


predictions = random_forest_classifier.predict(preproced_test_data)
predictions.shape


# In[ ]:


Columns = ['PassengerId', 'Survived']
rows = [[r['PassengerId'], predictions[i-1]] for i, r in test.iterrows()]
predictions_df = pd.DataFrame(np.array(rows), columns=Columns)
predictions_df.to_csv(index=False)


# ## With Neural Nets

# In[ ]:


import tensorflow as tf


# In[ ]:


activation_fn='elu'


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(7,),activation=activation_fn),
    tf.keras.layers.Dense(32, activation=activation_fn),
    tf.keras.layers.Dense(16, activation=activation_fn),
    tf.keras.layers.Dense(8, activation=activation_fn),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='Adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])


# In[ ]:


model.fit(preprocesed_training_set, y_train, epochs=10)


# In[ ]:


tf_predictions = model.predict(preproced_test_data)
tf_predictions = [1 if p>0.5 else 0 for p in tf_predictions]


# In[ ]:


tf_predictions


# In[ ]:


Columns = ['PassengerId', 'Survived']
rows = [[r['PassengerId'], tf_predictions[i-1]] for i, r in test.iterrows()]
predictions_df = pd.DataFrame(np.array(rows), columns=Columns)
predictions_df.to_csv(index=False)

