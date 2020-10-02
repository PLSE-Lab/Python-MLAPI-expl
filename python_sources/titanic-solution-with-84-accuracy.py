#!/usr/bin/env python
# coding: utf-8

# [![](https://cdn.iconscout.com/icon/free/png-256/github-109-438058.png)](https://github.com/gurusabarish/Titanic-prediction)
# # Train dataset: [https://github.com/gurusabarish/Titanic-prediction/blob/master/train.csv](https://github.com/gurusabarish/Titanic-prediction/blob/master/train.csv)
# # Test dataset: [https://github.com/gurusabarish/Titanic-prediction/blob/master/test.csv](https://github.com/gurusabarish/Titanic-prediction/blob/master/test.csv)
# > Github profile: [https://github.com/gurusabarish](https://github.com/gurusabarish)

# # **Import dataset**

# In[ ]:


#import the pandas library.
#we need to handle with arrays. So, we import numpy library as well.
import pandas as pd
import numpy as np

train = pd.read_csv('https://raw.githubusercontent.com/gurusabarish/Titanic-prediction/master/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/gurusabarish/Titanic-prediction/master/test.csv')


# # Analize the dataset

# > # Train data

# In[ ]:


#Print the first few rows of train dataset.
train.head()


# In[ ]:


# we dont use name and Ticket columns. So, we can drop those columns.
train.drop(['Name', 'Ticket'],axis=1, inplace=True)


# In[ ]:


#find the number of rows and columns
train.shape


# In[ ]:


#print the columns with his datatype and non-null(filled) counts.
train.info()


# > # Test data

# In[ ]:


# We should few columns in test data as well as train dta
test.drop(['Name', 'Ticket'],axis=1, inplace=True)


# # Visualization

# In[ ]:


#import libraries
# we will find "how many peoples are have the same value in respective column?"

import matplotlib.pyplot as plt
train.hist(figsize=(20,10), color='maroon', bins=20)
plt.show()


# # Handle the missing data

# > # Train data

# In[ ]:


# Count of missing data in each columns
train.isnull().sum()


# In[ ]:


# Find the percentage of missing data with respective columns.
columns = train.columns
for i in columns:
    percentage = (train[i].isnull().sum()/891)*100
    print(i,"\t %.2f" %percentage)


# In[ ]:


#Cabin has almost 80% missing data.So, we are drop the column.
train.drop(['Cabin'], axis=1, inplace=True)


# In[ ]:


#Embrked column has only few NAN value.So, drop the row which is contain the missing data.
train.dropna(subset=['Embarked'], inplace=True)


# In[ ]:


# Fill the NAN values with mean in Age column
train['Age'].fillna(train['Age'].mean(), inplace=True)


# In[ ]:


# Check the columns are don't contain the NAN values.
train.isnull().any()


# > # Test data

# In[ ]:


test.drop(['Cabin'], axis=1, inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


test['Age'].fillna(train['Age'].mean(), inplace=True)
test['Fare'].fillna(train['Fare'].mean(), inplace=True)


# In[ ]:


test.isnull().any()


# # Change data into numerical

# > # Train data

# In[ ]:


# Import the function from sklearn library
from sklearn import preprocessing
label = preprocessing.LabelEncoder()


# In[ ]:


# Find out the columns which is have object data type and store the column names.
object_columns = []
for i in train.columns:
    if train[i].dtype==object:
        object_columns.append(i)
print(object_columns)


# In[ ]:


# Convert the columns's object data to numerical data.
for i in object_columns:
    train[i] = label.fit_transform(train[i])


# In[ ]:


# Verify the dataset is contain only numerical values.
train.info()


# > # Test data

# In[ ]:


test_object_columns = []
for i in test.columns:
    if test[i].dtype==object:
        test_object_columns.append(i)
print(test_object_columns)


# In[ ]:


for i in test_object_columns:
    test[i] = label.fit_transform(test[i])


# In[ ]:


test.info()


# # Find the best algorithm

# > # Cross valitation

# In[ ]:


x = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


cross_val_score(LogisticRegression(), x, y).mean()


# In[ ]:


cross_val_score(SVC(), x, y).mean()


# In[ ]:


cross_val_score(RandomForestClassifier(), x, y).mean()


# In[ ]:


cross_val_score(GaussianNB(), x, y).mean()


# In[ ]:


cross_val_score(DecisionTreeClassifier(), x, y).mean()


# > # From cross validation, We can say randomforest is the best model for our dataset.

# # Train and test split

# In[ ]:


# Split the train dataset to build model.
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.30, random_state=5)


# # Find the best parameters for our model using grid search

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score
def evaluate(model, x_test, y_test):
  prediction = model.predict(x_test)
  print(accuracy_score(y_test, prediction))


# In[ ]:


from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [4,6,8,10,12],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1, 3, 4, 5],
    'min_samples_split': [1, 2, 4, 5],
    'n_estimators': [100, 200, 300, 1000]
}# Create a based model
rf = RandomForestClassifier()# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(x_train, y_train)
best = grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# # Random forest model

# In[ ]:


model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=12, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
model.fit(x_train,y_train)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score

predictions = model.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))


# # Evaluation

# In[ ]:


test.head()


# In[ ]:


survived=model.predict(test)
survived


# In[ ]:


ids = test['PassengerId']
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': survived })
output.head()


# In[ ]:


output.to_csv('output.csv', index=False)

