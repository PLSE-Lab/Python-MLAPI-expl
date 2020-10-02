#!/usr/bin/env python
# coding: utf-8

# # Regression Models

# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#from sklearn.neural_network import MLPRegressor
from math import sqrt


# ### Reading the data

# In[ ]:


data = pd.read_csv('../input/winequality-red.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# ### Selecting the input and output features for regression tasks

# In[ ]:


features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']


# In[ ]:


target = ['quality']


# #### Checking for any null values in the dataset

# In[ ]:


data.isnull().any()


# In[ ]:


X = data[features]
y = data[target]


# ### Perform train test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=200)


# ### Fit on train set

# In[ ]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ### Predict on test data

# In[ ]:


y_prediction = regressor.predict(X_test)
print(y_prediction[:5])
print('*'*40)
print(y_test[:5])


# In[ ]:


y_test.describe()


# ### Evaluate Linear Regression accuracy using root-mean-square-error

# In[ ]:


RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)


# ### 2. Decision Tree: Fit a new regression model to the traing set

# In[ ]:


regressor = DecisionTreeRegressor(max_depth=50)
regressor.fit(X_train, y_train)


# ### Perform prediction using decision tree regressor

# In[ ]:


y_prediction = regressor.predict(X_test)
y_prediction[:5]


# In[ ]:


y_test[:5]


# ### Evaluate Decision Tree Regression accuracy using root-mean-square-error

# In[ ]:


RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))


# In[ ]:


print(RMSE)


# When comparing two or more regression models, then the model with small `RMSE` will be better.

# # Classification Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# #### Copying the dataset to a new dataset so there will be no changes in the old dataset

# In[ ]:


data_classifier = data.copy()


# In[ ]:


data_classifier.head()


# In[ ]:


data_classifier['quality'].dtype


# ### Convert to a Classification Task

# Next we shall create a new column called Quality Label. This column will contain the values of 0 & 1
# 
# 1 <- good,
# 0 <- bad

# In[ ]:


data_classifier['quality_label'] = (data_classifier['quality'] > 6.5)*1


# In[ ]:


data_classifier['quality_label']


# ### Selecting the input and output features for classification tasks

# In[ ]:


features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']


# In[ ]:


X = data_classifier[features]
y = data_classifier[target_classifier]


# ### Perform train and test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# ### Fit on train set

# In[ ]:


wine_quality_classifier = DecisionTreeClassifier(max_leaf_nodes=20, random_state=0)


# In[ ]:


wine_quality_classifier.fit(X_train, y_train)


# ### Predict on test data

# In[ ]:


prediction = wine_quality_classifier.predict(X_test)
print(prediction[:5])
print('*'*10)
print(y_test['quality_label'][:5])


# ### Measure accuracy of the classifier

# In[ ]:


accuracy_score(y_true=y_test, y_pred=prediction)


# ## 2. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


data_classifier.head()


# ### Selecting the input and output features for classification tasks

# In[ ]:


features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']


# In[ ]:


X = data_classifier[features]
y = data_classifier[target_classifier]


# ### Perform train and test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# ### Fit on train set

# In[ ]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)


# ### Predict on test data

# In[ ]:


prediction = logistic_regression.predict(X_test)
print(prediction[:5])
print(y_test[:5])


# ### Measure accuracy of the classifier

# In[ ]:


accuracy_score(y_true=y_test, y_pred=prediction)


# In[ ]:




