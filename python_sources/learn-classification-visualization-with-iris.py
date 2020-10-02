#!/usr/bin/env python
# coding: utf-8

# # 1. Loading Libraries

# In[ ]:


import numpy as np
import pandas as pd
import random as rnd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split


# # **2. Reading Data**

# In[ ]:


df=pd.read_csv('../input/Iris.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


species_map={'Iris-versicolor':0,'Iris-setosa':1,'Iris-virginica':2}
df['Species']=df['Species'].replace(species_map)
df.head()


# In[ ]:


train=df.sample(frac=0.7,random_state=200)
test=df.drop(train.index)


# In[ ]:


train=train.reset_index()
train=train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']]
train.info()


# In[ ]:


test=test.reset_index()
test=test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']]
test.info()


#  **As till now , we have divided our dataset into test and train.Now we will work on train dataset only.**

# In[ ]:


train.head()


# In[ ]:


train.groupby(['Species']).count()


# # 3. Data Analysis & Visualisation

# In[ ]:


sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))
sns.boxplot(data=train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], orient="h");


# In[ ]:


sns.violinplot(x=train['Species'], y=train['SepalLengthCm']);


# In[ ]:


sns.violinplot(x=train['Species'], y=train['SepalWidthCm']);


# In[ ]:


sns.violinplot(x=train['Species'], y=train['PetalLengthCm']);


# In[ ]:


sns.violinplot(x=train['Species'], y=train['PetalWidthCm']);


# In[ ]:


sns.FacetGrid(train, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()


# In[ ]:


sns.FacetGrid(train, hue="Species", size=5).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()


# # **4. Various Models and Predictions**

# ## 4.1 Splitting Dataset in train and test

# In[ ]:


train.columns


# In[ ]:


X_train=train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#X_train=train[[ 'PetalLengthCm', 'PetalWidthCm']]

Y_train=train['Species']

X_test=test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#X_test=test[[ 'PetalLengthCm', 'PetalWidthCm']]

Y_test=test['Species']

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# ## 4.2 Modelling

# **LOGISTIC REGRESSION**

# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_test,Y_test) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
#coeff_df


# **We can see that length in directly related to corresponding classes while width is inversely related.**

# **SUPPORT VECTOR MACHINE**

# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_test,Y_test) * 100, 2)
acc_svc


# **KNN**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_test,Y_test) * 100, 2)
acc_knn


# **NAIVE BAYES**

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_test,Y_test) * 100, 2)
acc_gaussian


# **PERCEPTRON**

# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_test,Y_test) * 100, 2)
acc_perceptron


# **LINEAR SVC**

# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_test,Y_test) * 100, 2)
acc_linear_svc


# **SGD**

# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_test,Y_test) * 100, 2)
acc_sgd


# **DECISION TREE**

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test,Y_test) * 100, 2)
acc_decision_tree


# **RANDOM FOREST**

# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_test,Y_test) * 100, 2)
acc_random_forest


# ## 4.3 Comparison of Models

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Test Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Test Score', ascending=False)


# # Accuracy=95.56%

# # KNN is giving the best accuracy for this dataset.
