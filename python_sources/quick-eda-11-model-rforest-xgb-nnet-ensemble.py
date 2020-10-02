#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This Python 3 environment comes with many helpful analytics libraries installed
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


# importing the necessary libraries

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import seaborn as sns # for making plots with seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import ensemble


# In[ ]:


# Read the data
X = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')
X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')
X_sample = pd.read_csv('/kaggle/input/learn-together/sample_submission.csv', index_col='Id')


# In[ ]:


X.head()


# just looking quick at the training data, looks like they are all elready encoded in numbers so there is no need to do categorical conversion to numbers using label encoding or one hot encoding!

# In[ ]:


X.describe()


# from the above .describe() looks like there is no categorical variable that need to converted into number so we don't have to worry about it. Just to be sure we are checking it one more time:

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of forest categories(Target Variable)")
ax = sns.distplot(X["Cover_Type"])


# In[ ]:


# Get list of categorical variables
s = (X.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# wallaa, there is none of categorical variable!

# just a preprocessing stuff: removing missing target which is Cover_Type, Separating the target answer on separate placeholder: y, and dropping the target answer from the training dataset

# In[ ]:


# Remove rows with missing target, create a target dataset and remove Target from X dataset
X.dropna(axis=0, subset=['Cover_Type'], inplace=True)
y = X.Cover_Type              
X.drop(['Cover_Type'], axis=1, inplace=True)


# Calling the train_test_split algorithm for the training and validation purposes and putting the training and validation dataset on the proper placeholders

# In[ ]:


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)


# Calling the RandomForestClassifier algorithm since this is classification problem NOT a regression (number problem), and train the model using the training and target dataset

# In[ ]:


# Define the model
my_model_1 = RandomForestClassifier()

# Fit the model or training your model
my_model_1.fit(X_train, y_train)


# So after we trained the model, now we want to know how good is the model is. So how do we do that? We need a way to measure how good is the model is to 'unseen' data. For that we are calling mean_absolute_error to compute how 'good' our model is

# In[ ]:


# Get predictions or get the target output based on 'unseen' data in this case validation data.
predictions_1 = my_model_1.predict(X_valid)

# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_1)


# so we got 0.43, low, high ..i am not sure ? but what i know the lower the score the better it is the model since this is computing error not accuracy.

# So for the sake of fun we will try another metrics, accuracy ... so ideally the higher the better it is

# In[ ]:



accuracy = accuracy_score(y_valid, predictions_1)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Trying for the second model with different parameters

# In[ ]:


# Define the model
my_model_2 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

# Fit the model
my_model_2.fit(X_train, y_train) 

# Get predictions
predictions_2 = my_model_2.predict(X_valid) 

# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid) 

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)

accuracy = accuracy_score(y_valid, predictions_2)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Turns out to be worst than the first model! Why is that? I think the n_estimators is too low! Let's try a bigger n_estimators!

# In[ ]:


# Define the model
my_model_3 = RandomForestClassifier(n_estimators=1000, max_depth=20,random_state=0)

# Fit the model
my_model_3.fit(X_train, y_train)

# Get predictions
predictions_3 = my_model_3.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(predictions_3, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, predictions_3)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# woow, this third model looks good! Let's try again, this time with XGBClassifier() but the same n_estimators as RandomForestClassifier()

# In[ ]:


# Basic XGB classifier
model_4 = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=20,random_state=0)
model_4.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds4_test = model_4.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds4_test, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds4_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# let's try with max_depth = 5

# In[ ]:


# Basic XGB classifier
model_5 = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=5,random_state=0)
model_5.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds5_test = model_5.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds5_test, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds5_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# let's try with max_depth = 10

# In[ ]:


# Basic XGB classifier
model_6 = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=10,random_state=0)
model_6.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds6_test = model_6.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds6_test, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds6_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# let's try with max_depth = 15

# In[ ]:


# Basic XGB classifier
model_7 = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=15,random_state=0)
model_7.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds7_test = model_7.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds7_test, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds7_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# OK, let's try max_depth = 25

# In[ ]:


# Basic XGB classifier
model_8 = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=25,random_state=0)
model_8.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds8_test = model_8.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds8_test, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds8_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


model_9 = ensemble.ExtraTreesClassifier(n_estimators=350)
model_9.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds9_test = model_9.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds9_test, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds9_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


trying using neural network with KERAS API


# In[ ]:


X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[ ]:


from tensorflow.python.data import Dataset
import tensorflow as tf
from tensorflow import keras


# In[ ]:


model_10 = keras.Sequential([
 keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(X_train.shape[1],)), # neurons with relu activation, first layer with input 
 #keras.layers.Dropout(0.5), # dropout for reducing the overfitting problem
 #keras.layers.Dense(512, activation=tf.nn.relu), # 2nd hidden layer
 #keras.layers.Dropout(0.5),
 #keras.layers.Dense(256, activation=tf.nn.relu), # 3rd hidden layer
 #keras.layers.Dropout(0.5),
 keras.layers.Dense(7, activation=tf.nn.softmax)]) #  output layer with 7 categories

model_10.compile(loss='sparse_categorical_crossentropy', #this loss method is useful for multiple categories, otherwise our model does not work
 optimizer=tf.train.AdamOptimizer(learning_rate=0.0043, beta1=0.9), metrics=['accuracy'])


# In[ ]:


# train the model
history1 = model_10.fit(X_train, y_train, epochs = 300, batch_size = 32, verbose=2, validation_data = (X_valid, y_valid))


# this is interesting, why is it not working with neural network? is it too small the dataset ... my guess is.

# In[ ]:


model_9 = ensemble.ExtraTreesClassifier(n_estimators=350)
model_9.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds9_test = model_9.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds9_test, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds9_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# we will try a stacking method this time!

# In[ ]:


import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_hist_gradient_boosting
# Going to use these  base models for the stacking
from sklearn.ensemble import (  BaggingClassifier, ExtraTreesClassifier,  HistGradientBoostingClassifier)
from tqdm import tqdm
from mlxtend.classifier import StackingCVClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


clf_lgbm = LGBMClassifier(n_estimators=400,num_leaves=100,verbosity=0)
clf_knc = KNeighborsClassifier(n_jobs = -1, n_neighbors =1)
clf_etc = ExtraTreesClassifier(random_state = 1, n_estimators = 900, max_depth =50,max_features = 30)
clf_hbc = HistGradientBoostingClassifier(random_state = 1, max_iter = 500, max_depth =25)


# In[ ]:


ensemble = [
            ('clf_knc', clf_knc),
            ('clf_hbc', clf_hbc),
            ('clf_etc', clf_etc),
            ('clf_lgbm', clf_lgbm)
           
           ]

model_stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],
                             meta_classifier=clf_lgbm,
                             cv=3,
                             use_probas=True, 
                             use_features_in_secondary=True,
                             verbose=-1,
                             n_jobs=-1)


# In[ ]:


model_stack.fit(X_train, y_train)

# Preprocessing of test data, fit model
preds_stack = model_stack.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(preds_stack, y_valid)

#print MAE
print("Mean Absolute Error:" , mae_3)

accuracy = accuracy_score(y_valid, preds_stack)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# So looks like the stacking model gives the best result! So, I am going to use the stacking model for prediction and submit the result for submission!

# Computing the predictions using the test dataset

# In[ ]:


preds_test = model_stack.predict(X_test)


# Now organising those predictions with ID and saving it out the submission file

# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'Cover_Type': preds_test})
output.to_csv('submission.csv', index=False)


# In[ ]:




