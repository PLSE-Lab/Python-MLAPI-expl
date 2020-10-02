#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <img src="https://i.imgur.com/qBl2QL2.jpg" width="600px">

# I made this kernel for Kaggle's Titanic competition (and it also happens to be my first one!). In this kernel, given some information, I tried to predict whether a given person aboard the ship had survived.

# # Contents

# * Preliminary steps
#     * Importing the necessary libraries
#     * Converting the CSV file into a pandas dataframe
# * Encoding the features of the train data
# * Defining the features and prediction target
# * Creating the model
# * Fitting the model
# * Encoding the features of the test data
# * Predicting survival
# * Ending Note

# ### Preliminary Steps

# Importing the necessary libraries -

# In[ ]:


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Input

tqdm.pandas()
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Converting the CSV file into a pandas dataframe -

# In[ ]:


train_path = '/kaggle/input/titanic/train.csv'
train_data = pd.read_csv(train_path)
train_data = train_data.fillna(train_data.mean())


# A look at some of the train data - 

# In[ ]:


train_data.head(10)


# ### Encoding the features of the train data

# Encoding the 'Sex' and 'Embarked' features - 

# In[ ]:


def process_sex(x):
    if x == "male":
        return 1
    else:
        return 0
    
def process_embarked(x):
    code = [0, 0, 0, 0]
    ports = ["C", "Q", "S"]

    if x in ports:
        code[list.index(ports, x)] = 1
    else:
        code[-1] = 1
        
    return tuple(code)
        
train_data["Sex"] = train_data["Sex"].progress_apply(process_sex)
train_data["Embarked"] = train_data["Embarked"].progress_apply(process_embarked)


# Splitting the lists of numbers under the feature 'Embarked' to obtain 4 different columns containing data of only one number per each row - 

# In[ ]:


train_data["Embarked_0"] = [train_data["Embarked"][idx][0] for idx in tqdm(range(len(train_data)))]
train_data["Embarked_1"] = [train_data["Embarked"][idx][1] for idx in tqdm(range(len(train_data)))]
train_data["Embarked_2"] = [train_data["Embarked"][idx][2] for idx in tqdm(range(len(train_data)))]
train_data["Embarked_3"] = [train_data["Embarked"][idx][3] for idx in tqdm(range(len(train_data)))]


# A look at the encoded features - 

# In[ ]:


train_data.head(10)


# ### Defining the features and prediction target - 

# In[ ]:


y = train_data["Survived"].values.reshape(len(train_data), 1)
X = train_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_0", "Embarked_1", "Embarked_2", "Embarked_3"]].values


# Bringing all the features to a range between 0 and 1 by dividing all the values of a feature by its biggest value - 

# In[ ]:


X = X/X.max(axis=0)


# Splitting the training data into training data and validation data - 

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y)


# ### Creating the model

# In[ ]:


model = Sequential()
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# Providing the input size to the model - 

# In[ ]:


model.build(input_shape=(None, 10))
model.summary()


# ### Fitting the model - 

# In[ ]:


model.fit(x=train_X, y=train_y, validation_data=(val_X, val_y), epochs=10)


# ### Encoding the features of the test data

# Converting the CSV file into a pandas dataframe -

# In[ ]:


test_path = '/kaggle/input/titanic/test.csv'
test_data = pd.read_csv(test_path)
test_data = test_data.fillna(test_data.mean())


# A look at the test data - 

# In[ ]:


test_data.head(10)


# Encoding the 'Sex' and 'Embarked' features - 

# In[ ]:


test_data["Sex"] = test_data["Sex"].progress_apply(process_sex)
test_data["Embarked"] = test_data["Embarked"].progress_apply(process_embarked)


# A look at the encoded features - 

# In[ ]:


test_data


# Splitting the lists of numbers under the feature 'Embarked' to obtain 4 different columns containing data of only one number per each row - 

# In[ ]:


test_data["Embarked_0"] = [test_data["Embarked"][idx][0] for idx in tqdm(range(len(test_data)))]
test_data["Embarked_1"] = [test_data["Embarked"][idx][1] for idx in tqdm(range(len(test_data)))]
test_data["Embarked_2"] = [test_data["Embarked"][idx][2] for idx in tqdm(range(len(test_data)))]
test_data["Embarked_3"] = [test_data["Embarked"][idx][3] for idx in tqdm(range(len(test_data)))]


# Defining a new variable to hold the features - 

# In[ ]:


X_test = test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_0", "Embarked_1", "Embarked_2", "Embarked_3"]].values


# Bringing all the features to a range between 0 and 1 by dividing all the values of a feature by its biggest value. The second line of code in the cell below deals with cases where the biggest value of the feature happens to be 0. Since you cannot divide by 0, it simply substitutes 0 instead.

# In[ ]:


X_test = X_test/X_test.max(axis=0)
X_test[np.isnan(X_test)] = 0


# ### Predicting Survival

# Running inference on the test data and rounding it off to either 0 or 1 - 

# In[ ]:


predictions = model.predict(X_test)
predictions = np.round(predictions).reshape(len(X_test))


# Since gender_submission.csv is of the format in which our submission is supposed to be made, I'm first importing it and converting it into a pandas dataframe - 

# In[ ]:


sub_path = '/kaggle/input/titanic/gender_submission.csv'
submission = pd.read_csv(sub_path)


# A look at gender_submission.csv - 

# In[ ]:


submission.head(10)


# Replacing the 'Survived' column in the dataframe with the values we got - 

# In[ ]:


submission["Survived"] = np.int32(predictions)


# A final look at the dataframe with our predictions - 

# In[ ]:


submission.head(10)


# Converting the dataframe into a csv file without the index column - 

# In[ ]:


submission.to_csv('submission.csv', index=False)


# ## Ending Note

# This being my first ML model, I learnt basics like how to create a neural network and encode features. I really enjoyed it, and look forward to learning more in the future. I also really appreciate feedback to help me improve both the accuracy and efficiency of my model :)
