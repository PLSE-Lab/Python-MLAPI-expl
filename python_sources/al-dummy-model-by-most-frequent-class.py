#!/usr/bin/env python
# coding: utf-8

# Dummy Classification Model

# In[ ]:


# importing train and test data into train_df and test_df dataframes
import pandas as pd
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


# train and test dataframes column names
print(train_df.columns.values)
print('=' * 40)
print(test_df.columns.values)


# In[ ]:


# preview the train data
train_df.head(n=4)
train_df.tail(n=7)


# In[ ]:


# preview the test data
test_df.head()
train_df.tail()


# In[ ]:


# train and test features data types
train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


# numerical features distribution
print(train_df.describe())
print('_'*40)
print(test_df.describe())


# In[ ]:


# categorical features distribution
print(train_df.describe(include=['O']))
print('_'*80)
print(test_df.describe(include=['O']))


# In[ ]:


# preparing training data
train_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','Survived']
train_data = train_df[train_cols]


# In[ ]:


# retrieving 7 first rows from data frame
train_data.head(n=7)                            


# In[ ]:


# retrieving 7 last rows from data frame
train_data.tail(n=7)                            


# In[ ]:


# get amount of missing data
train_data.isnull().sum()


# In[ ]:


# massaging training data
train_data_m = train_data.copy(deep = True)     # dataframe deep copy
train_data_m = train_data_m.dropna()            # train data frame missing values imputation
x_train_data_m =  train_data_m.drop('Survived', axis = 1)                  # removing column from dataframe
y_train_data_m = train_data_m['Survived']       


# In[ ]:


# majority rule model
from sklearn.dummy import DummyClassifier
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(x_train_data_m, y_train_data_m)
dummy_model_train_prediction = dummy_model.predict(x_train_data_m) 


# In[ ]:


# estimating model accuracy on training data
from sklearn.metrics import accuracy_score
dummy_model_train_prediction_accuracy = round(accuracy_score(y_train_data_m, dummy_model_train_prediction)*100,2)
print(dummy_model_train_prediction_accuracy,'%')                              


# In[ ]:


# preparing testing data
test_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
test_data = test_df[test_cols]
test_data.info()


# In[ ]:


# massaging testing data
test_data_m = test_data.copy(deep = True)
test_data_m['Age'].fillna((test_data_m['Age'].mean()), inplace=True)
test_data_m['Fare'].fillna((test_data_m['Fare'].mean()), inplace=True)
test_data_m.info()


# In[ ]:


# preparing submission data
ID = test_df['PassengerId']
P = dummy_model.predict(test_data_m)


# In[ ]:


# preparing submission file
submission = pd.DataFrame( { 'PassengerId': ID , 'Survived': P } )
submission.to_csv('dummy_model_v1.csv' , index = False )


# In[ ]:


import pandas as pd
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

