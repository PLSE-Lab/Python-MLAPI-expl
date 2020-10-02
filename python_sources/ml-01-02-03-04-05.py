#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly as py
from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs
import plotly.graph_objs as go

init_notebook_mode(connected=True)
import plotly.offline as offline


# In[ ]:


print("""





EXERCISE 01 : 
Use the dataset IRIS of scikit-learn to illustrate how to use the KNN algorithm with
splitting data.






""")


# In[ ]:


import os
os.listdir("../input/iris-dataset")


# In[ ]:


iris = pd.read_csv("../input/iris-dataset/iris.csv")
iris.fillna(method = "bfill", inplace = True)
iris.head()


# In[ ]:


iris.info()


# In[ ]:


iris.describe()


# In[ ]:


sns.pairplot(data = iris, hue='species')


# In[ ]:


"""Model Preparation"""


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris["species"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))
print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


df = pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted', index=X_test.index)], ignore_index=False, axis=1)


# In[ ]:


df.head()


# In[ ]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:


print("""




EXERCISE 02:
Use the dataset IRIS of scikit-learn to illustrate how to use pipeline, scaling, grid
search and the KNN algorithm.





""")


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection  import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


# In[ ]:


X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


# normilaizer data
# classification
pipeline = Pipeline([
    ('normalizer', StandardScaler()), 
    ('clf', LogisticRegression()) 
])


# In[ ]:


scores = cross_validate(pipeline, X_train, y_train)
scores


# In[ ]:


scores["test_score"].mean()


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[ ]:


k_range=range(1,31)
param_grid=dict(n_neighbors=k_range)
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')


# In[ ]:


grid.fit(X = X_train,y = y_train)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


# In[ ]:


weight_options=['uniform','distance']
param_grid=dict(n_neighbors=k_range,weights=weight_options)
print (param_grid)


# In[ ]:


grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


print (grid.best_score_)
print (grid.best_params_)


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=13,weights='uniform')
knn.fit(X_train,y_train)
pre = knn.predict(X_test)


# In[ ]:


knn.score(X_test, y_test)


# In[ ]:


X_test["species"] = y_test
X_test["prediction"] = pre
X_test.head()


# In[ ]:


print("""





EXERCISE 03:
Use the dataset IRIS of scikit-learn to illustrate how to use the logistic regression
algorithm with splitting data.






""")


# In[ ]:


X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris["species"]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# In[ ]:


logreg.score(X_test, y_test)


# In[ ]:


X_test["species"] = y_test
X_test["prediction"] = y_pred
X_test.head()


# In[ ]:


print("""


EXERCISE 05:
Illustrate how to use the linear regression algorithm with splitting data on the Boston
housing dataset.


""")


# In[ ]:


bs = pd.read_csv("../input/boston-house/boston.csv")
bs.head()


# In[ ]:


plt.figure(figsize=(20,10)) 
sns.heatmap(bs.corr(),annot=True,cmap='cubehelix_r') 
plt.show()


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


X = bs["RM"].values.reshape(-1, 1)
y = bs["MEDV"].values.reshape(-1, 1)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=101)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


plt.figure(figsize = (20, 10))
plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train), color =  "red")
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.title("Trainning Set : RM vs MEDV")


# In[ ]:


plt.figure(figsize = (20, 10))
plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color =  "red")
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.title("Test Set : RM vs MEDV")


# In[ ]:


X = bs["LSTAT"].values.reshape(-1, 1)
y = bs["MEDV"].values.reshape(-1, 1)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=101)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


plt.figure(figsize = (20, 10))
plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train), color =  "red")
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.title("Trainning Set : LSTAT vs MEDV")


# In[ ]:


plt.figure(figsize = (20, 10))
plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color =  "red")
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.title("Test Set : LSTAT vs MEDV")


# In[ ]:





# In[ ]:





# In[ ]:




