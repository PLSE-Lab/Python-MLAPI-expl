#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Assignment</h1>
# <h2 align="center">Faisal Akhtar</h2>
# <h2 align="center">Roll No.: 17/1409</h2>
# <p>Machine Learning - B.Sc. Hons Computer Science - Vth Semester</p>
# <p>Perform logistic regression for classification on iris dataset. Apply ridge regularization and compare the performance before and after regularization.</p>

# ### Libraries Imported

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn import metrics


# ### Loading IRIS dataset from Scikit-Learn's dataset

# In[ ]:


from sklearn.datasets import load_iris
iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
data['species'] = iris_data.target

print(data.head())


# In[ ]:


X = data.drop('species',axis=1)
Y = data['species']
print("X = ",X.head())
print("Y = ",Y.head())


# ### Test/Train Split
# <p>Dividing data into test-train sets, 30% and 70%</p>

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# ### Logistic Regression
# <p>Fit the model according to "data" variable obtained from CSV.</p>

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)


# **Logistic Regression model metrics**

# In[ ]:


print('Accuracy score: ', metrics.accuracy_score(Y_test, Y_pred))
print('Precision score: ', metrics.precision_score(Y_test, Y_pred, average='micro'))
print('Recall score: ', metrics.recall_score(Y_test, Y_pred, average='micro'))
print('F1 score: ', metrics.f1_score(Y_test, Y_pred, average='micro'))
print('Confusion Matrix :\n', metrics.confusion_matrix(Y_test, Y_pred))


# ### Ridge Regression
# <p>Higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely restricted, in this case linear and ridge regression resembles</p>

# In[ ]:


rr = Ridge(alpha=0.5)
rr.fit(X_train, Y_train)

Y_predRR = rr.predict(X_test)


# In[ ]:


print('Accuracy score: ', metrics.accuracy_score(Y_test, Y_predRR))
print('Precision score: ', metrics.precision_score(Y_test, Y_predRR, average='micro'))
print('Recall score: ', metrics.recall_score(Y_test, Y_predRR, average='micro'))
print('F1 score: ', metrics.f1_score(Y_test, Y_predRR, average='micro'))
print('Confusion Matrix :\n', metrics.confusion_matrix(Y_test, Y_predRR))


# ### Before and after metrics
# Comparing the performance before and after applying L2 regularization.

# In[ ]:


train_score=lr.score(X_train, Y_train)
test_score=lr.score(X_test, Y_test)

Ridge_train_score = rr.score(X_train, Y_train)
Ridge_test_score = rr.score(X_test, Y_test)


# In[ ]:


print("Logistic regression train score:", train_score)
print("Logisitic regression test score:", test_score)
print("Ridge regression train score:", Ridge_train_score)
print("Ridge regression test score:", Ridge_test_score)

