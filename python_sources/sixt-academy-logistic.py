#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression on titanic dataset

# ## Table of Contents
# 
# 1. [Problem Statement](#section1)<br>
# 2. [Data Loading and Description](#section2)
# 3. [Preprocessing](#section3)
# 4. [Logistic Regression](#section4)<br>
#     - 4.1 [Preparing X and y using pandas](#section401)<br>
#     - 4.2 [Splitting X and y into training and test dataset](#section402)<br>
#     - 4.3 [Logistic regression in scikit-learn](#section403)<br>
#     - 4.4 [Using the Model for Prediction](#section404)<br>
# 5. [Model evaluation](#section5)<br>

# <a id='section1'></a>
# ### 1. Problem Statement

# The goal is to __predict survival__ of passengers travelling in RMS __Titanic__ using __Logistic regression__.

# <a id='section2'></a>
# ### 2. Data Loading and Description

# - The dataset consists of the information about people boarding the famous RMS Titanic. Various variables present in the dataset includes data of age, sex, fare, ticket etc. 
# - The dataset comprises of __891 observations of 9 columns__. Below is a table showing names of all the columns and their description.

# | Column Name   | Description                                               |
# | ------------- |:-------------                                            :| 
# | PassengerId   | Passenger Identity                                        | 
# | Survived      | Whether passenger survived or not                         |  
# | Pclass        | Class of ticket                                           | 
# | Name          | Name of passenger                                         |   
# | Sex           | Sex of passenger                                          |
# | Age           | Age of passenger                                          |
# | SibSp         | Number of sibling and/or spouse travelling with passenger |
# | Parch         | Number of parent and/or children travelling with passenger|                                       
# | Fare          | Price of ticket                                           |

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# #### Importing the Dataset

# In[ ]:


import os
data = pd.read_csv('/kaggle/input/titanic_data.csv')


# In[ ]:


data.head(10)


# In[ ]:


data.shape


# In[ ]:


data.info()


# <a id='section3'></a>
# ## 3. Preprocessing the data

# In[ ]:


sns.boxplot(data['Age']);


# In[ ]:


(20+22+20+24+80)/5


# In[ ]:


20,20,22,24,80


# - Dealing with missing values<br/>
#     - Replacing missing values of __Age__ with median values.

# In[ ]:


data.Age.median()


# In[ ]:


data['Age'].fillna(data.Age.median(),inplace=True)


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data['family_size'] = data['SibSp']+data['Parch']+1


# In[ ]:


data.head()


# In[ ]:


cols_to_remove = ['PassengerId','Name','SibSp','Parch','Fare']
data.drop(cols_to_remove, axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data['Sex'].unique()


# In[ ]:


data_with_dummies = pd.get_dummies(data, columns=['Sex'],drop_first=True)


# In[ ]:


data_with_dummies.head()


# In[ ]:


sns.pairplot(data_with_dummies);


# <a id='section4'></a>
# ## 4. Logistic Regression

# <a id='section401'></a>
# ## 4.1 Preparing X and y using pandas

# In[ ]:


data_with_dummies.head()


# In[ ]:


data_with_dummies.columns


# In[ ]:


features = ['Pclass', 'Age', 'family_size', 'Sex_male']
target = ['Survived']


# In[ ]:


x = data_with_dummies[features]
y = data_with_dummies[target]


# In[ ]:


x.head()


# In[ ]:


y.head()


# <a id='section402'></a>
# ## 4.2 Splitting X and y into training and test datasets.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=96)


# In[ ]:


print('x-train shape', x_train.shape)
print('x-test shape', x_test.shape)
print('y-train shape', y_train.shape)
print('y-test shape', y_test.shape)


# <a id='section403'></a>
# ## 4.3 Logistic regression in scikit-learn

# In[ ]:


from sklearn.linear_model import LogisticRegression


# To apply any machine learning algorithm on your dataset, basically there are 4 steps:
# 1. Load the algorithm
# 2. Instantiate and Fit the model to the training dataset
# 3. Prediction on the test set
# 4. Evaluation of the model

# In[ ]:


log_reg = LogisticRegression()


# In[ ]:


log_reg.fit(x_train, y_train)


# <a id='section404'></a>
# ## 4.4 Using the Model for Prediction

# In[ ]:


y_pred = log_reg.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


log_reg.predict_proba(x_test)[:,1]


# In[ ]:


y_pred.sum()


# <a id='section5'></a>
# ## 5. Model evaluation 

# __Error__ is the _deviation_ of the values _predicted_ by the model with the _true_ values.<br/>
# We will use __accuracy score__ and __confusion matrix__ for evaluation.

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


pd.DataFrame(confusion_matrix(y_test, y_pred))


# In[ ]:


pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['predicted_not_survived','predicted_survived'], index=['actual_not_survived','actual_survived'])


# In[ ]:


(105+41)/(105+18+15+41)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


41/(41+18)


# In[ ]:





# 100
# 
# 90 negative
# 10 positive
# 
# 90%
# 0%

# In[ ]:


pd.DataFrame([[90,10],[0,0]], columns=['predicted_negative','predicted_positive'], index=['actual_negative','actual_positive'])


# classwise 
# 
# 100% accurate for negative cases ::
# 0% accurate for positive cases

# In[ ]:




