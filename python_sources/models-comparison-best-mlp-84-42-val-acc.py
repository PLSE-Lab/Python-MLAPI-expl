#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/HinePo/diabetes-ML-and-DL-study/blob/master/Diabetes_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Description

# In this kernel we will try to predict if a person has diabetes or not. We will do some preliminary modeling analysis and compare the results. It is contained: 
# 
# 1. Applications for some models
# 2. Plot accuracy from models
# 3. Training the best model
# 4. Predictions
# 
# Best model: MLP Classifier (84.42 % validation accuracy).
# 
# Dataset can be found in
# https://www.kaggle.com/uciml/pima-indians-diabetes-database

# # Importing

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-pastel")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix


# # Loading dataset

# In[ ]:


df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# # Defining features and target

# In[ ]:


all_vars = np.array(df.columns)
all_vars


# We will use the following eight variables to predict diabetes:
# 
# - Number of times pregnant
# - Glucose
# - Blood pressure
# - Skin thickness
# - Insulin
# - Body mass index
# - Diabetes pedigree function
# - Age

# In[ ]:


# features: columns the classifier will use to predict

features = np.array(all_vars[0:8])
features


# Outcome column will be the target/predicted variable.
# 
# Outcome = 0 : healthy
# 
# Outcome = 1 : diabetes predicted

# In[ ]:


# target: column we want to predict

target = np.array(all_vars[8])
target


# # Splitting the dataset

# In[ ]:


# split dataset using arrays as filters
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size = 0.2,
                                                      stratify = df[target], random_state = 0)


# In[ ]:


# Creating variables to store the results
all_models = np.array([])
all_scores = np.array([])


# In[ ]:


all_models


# # Modeling

# ## Support Vector Machine (SVM)

# Link for documentation:
# https://scikit-learn.org/stable/modules/svm.html

# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


def svm_test(X_train, y_train, cv = 10):
  np.random.seed(0)
  svc = LinearSVC()
  cv_scores = cross_val_score(svc, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()


# In[ ]:


res = svm_test(X_train, y_train)


# In[ ]:


# updating results
all_models = np.append(all_models, "SVM")
all_scores = np.append(all_scores, res)


# In[ ]:


all_models, all_scores


# ## Extra Trees Classifier

# Link for documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


def ext_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  ext = ExtraTreesClassifier(n_estimators = n_estimators, criterion = 'entropy', random_state = 0, n_jobs = -1)
  cv_scores = cross_val_score(ext, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()


# In[ ]:


res = ext_test(X_train, y_train)


# In[ ]:


# updating results
all_models = np.append(all_models, "ETC")
all_scores = np.append(all_scores, res)


# ## Random Forest Classifier

# Link for documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


def rfc_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  rfc = RandomForestClassifier(n_estimators = n_estimators, random_state = 0, n_jobs = -1)
  cv_scores = cross_val_score(rfc, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()


# In[ ]:


res = rfc_test(X_train, y_train)


# In[ ]:


# updating results
all_models = np.append(all_models, "RFC")
all_scores = np.append(all_scores, res)


# ## XGBClassifier

# Link for documentation: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


def xgbc_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  xgb = XGBClassifier()
  cv_scores = cross_val_score(xgb, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()


# In[ ]:


res = xgbc_test(X_train, y_train)


# In[ ]:


# updating results
all_models = np.append(all_models, "XGB")
all_scores = np.append(all_scores, res)


# ## Multi-Layer Perceptron (MLP)

# Link for documentation: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:


def mlp_test(X_train, y_train, cv = 10):
  np.random.seed(0)

  mlp = MLPClassifier()
  scaler = StandardScaler()

  pipe = Pipeline([('scaler', scaler), ('mlp', mlp)])

  cv_scores = cross_val_score(pipe, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv,  'tests: ', cv_scores.mean())
  return cv_scores.mean()


# In[ ]:


res = mlp_test(X_train, y_train)


# In[ ]:


# updating results
all_models = np.append(all_models, "MLP")
all_scores = np.append(all_scores, res)


# ## Keras Sequential Model

# Link for documentation: https://keras.io/guides/sequential_model/

# In[ ]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# define the keras model
model = Sequential()

# 8 input features (input_dim)
model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

# last layer must be activated with sigmoid or softmax since we want results in (0, 1) range (probabilities)
model.add(Dense(1, activation='sigmoid'))

# compile the keras model, choose metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=200, batch_size=10, verbose = 0)


# In[ ]:


# evaluate model
test_loss, res = model.evaluate(X_test, y_test)
round(test_loss, 4), round(res, 4)


# In[ ]:


# updating results
all_models = np.append(all_models, "Keras trained")
all_scores = np.append(all_scores, res)


# In[ ]:


# predict classes with the model
# class 0 : no diabetes 
# class 1 : diabetes predicted :(
predict_class = model.predict_classes(X_test)
predict_class[10:15]


# In[ ]:


# predict on test data
y_pred = model.predict(X_test)
y_pred[0:5]


# In[ ]:


# we will need this to calculate confusion matrix
rounded = [round(x[0]) for x in y_pred]
rounded[0:20]


# In[ ]:


# summarize the first n cases
n = 5
for i in range(n):
	print('%s => \n %d (expected %d)\n\n' % (X_test.iloc[i, ].tolist(), rounded[i], y_test.iloc[i]))


# In[ ]:


# input to confusion_matrix must be an array of int (rounded)
# obviously, we can only call confusion_matrix once we already called the fit method on the model
matrix = confusion_matrix(y_test, rounded)


# In[ ]:


matrix


# # Comparing Models

# In[ ]:


# check models and scores arrays
all_models, all_scores


# In[ ]:


# plot model results

fig, ax = plt.subplots()
ax.barh(all_models, all_scores)
plt.xlim(0, 1)
for index, value in enumerate(all_scores):
    plt.text(value, index, str(round(value, 4)), fontsize = 12)


# In[ ]:


best_model = all_models[all_scores.argmax()]


# In[ ]:


# this is just a string, it doesn't contain the model parameters
best_model


# # Training best model

# In[ ]:


# Defining model
mlp = MLPClassifier()

# using a scaler, since it is a neural network
scaler = StandardScaler()

# creating the pipeline with scaler and then MLP
pipe = Pipeline([('scaler', scaler), ('mlp', mlp)])


# In[ ]:


# fit/train the algorithm on the train data
pipe.fit(X_train, y_train)


# # Making predictions

# In[ ]:


# predict classes with the model
# class 0 : no diabetes 
# class 1 : diabetes predicted :(
y_pred = pipe.predict(X_test)
y_pred


# In[ ]:


pipe.predict_proba(X_train)


# In[ ]:


res = pipe.score(X_test, y_test)
res


# In[ ]:


# now that we trained (fit) the model, we can calculate the confusion matrix
cm = confusion_matrix(y_pred, y_test)
cm


# # Last results

# In[ ]:


# updating results (appending trained model)
all_models = np.append(all_models, "MLP trained")
all_scores = np.append(all_scores, res)


# In[ ]:


all_models, all_scores


# In[ ]:


# plot model results with trained model

fig, ax = plt.subplots()
ax.barh(all_models, all_scores)
plt.xlim(0, 1)
plt.title("Diabetes prediction: Model vs Accuracy")
for index, value in enumerate(all_scores):
    plt.text(value, index, str(round(value, 4)), fontsize = 12)

