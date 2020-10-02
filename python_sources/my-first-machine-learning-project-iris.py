#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# In[5]:


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pd.read_csv('../input/iris.csv')
dataset.columns
dataset.shape
dataset.head(20)
print(dataset.describe())
dataset.groupby('species').size()


# In[ ]:


Data Visualization


# The first step for data visualization is plotting individual variable or univariate plot.
# For this first I used **Boxplot**.

# 

# In[6]:


dataset.plot(kind='box',subplots=True,layout=(2,2))
plt.show()


# In[ ]:


Plotting using Histogram


# In[7]:


dataset.hist()
plt.show()


# **Density Plot** - It is similar to histogram.

# In[8]:


dataset.plot(kind='density',subplots=True,layout=(2,2))
plt.show()


# **Multivariate plotting** - In this we have to find relationship between the variables.

# Scatter Matrix - Its a matrix of scatter plots.

# In[9]:


from pandas.plotting import scatter_matrix
scatter_matrix(dataset,figsize=(10,9))
plt.show()


# Creating Training and Validation set

# In[10]:


X=dataset.iloc[:,:4]
y=dataset.iloc[:,4]
validation_size=0.2


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# Splitting Training and Validation set

# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=validation_size,random_state=2)


# **Model Building**

# In[13]:


models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=10,random_state=2)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg="%s %f %f"%(name,cv_results.mean(),cv_results.std())
    print(msg)


# **Algorithm Comparison**

# In[14]:


fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# **Prediction**

# In[16]:


knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# Decision tree

# In[18]:


clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)


# Predicting labels on test data

# In[19]:


y_pred=clf.predict(X_test)
print('Accuracy score  on train data:',accuracy_score(y_train,clf.predict(X_train)))
print('Accuracy score  on test data:',accuracy_score(y_test,y_pred))

print('Confusion matrix :',confusion_matrix(y_test,y_pred))
print('Classification report :',classification_report(y_test,y_pred))


# We will tune the parameters of the decision tree to increase its accuracy.
# With min_samples_split -> minimum number of samples required to split an internal node

# In[20]:


clf=DecisionTreeClassifier(criterion='entropy',min_samples_split=10)
clf.fit(X_train,y_train)


# Predicting labels on test data

# In[23]:


y_pred=clf.predict(X_test)


# Predicting Accuracy

# In[24]:


print('Accuracy score  on train data:',accuracy_score(y_train,clf.predict(X_train)))
print('Accuracy score  on test data:',accuracy_score(y_test,y_pred))

print('Confusion matrix :',confusion_matrix(y_test,y_pred))
print('Classification report :',classification_report(y_test,y_pred))

