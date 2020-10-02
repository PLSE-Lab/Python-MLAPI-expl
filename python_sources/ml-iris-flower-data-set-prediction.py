#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
import scipy
import os
print(os.listdir("../input"))


# **Loading Dataset (CSV file)**
# 

# In[ ]:


ir=pd.read_csv("../input/Iris.csv")


# **Summarizing Dataset**

# In[ ]:


ir.shape


# In[ ]:


ir.head(10)


# In[ ]:


ir1=ir.drop('Id',1)
ir1.head(10)


# In[ ]:


ir1.describe()


# In[ ]:


ir1.groupby('Species').size()


# In[ ]:


#Box Plot
ir1.plot(kind='box',subplots=False,layout=(2,2),sharex=False,sharey=False)
ir1.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)


# In[ ]:


#Histogram Chart
ir1.hist(figsize=(10,10))


# In[ ]:


#Density Chart
ir1.plot(kind='density',subplots=True,figsize=(6,6))


# In[ ]:


#Line Chart
ir1.plot(kind='line',subplots=True,figsize=(6,6))


# In[ ]:


species_table=pd.crosstab(index=ir1['Species'],columns="count")
species_table


# In[ ]:


#Bar Plot
species_table.plot(kind='bar')
plt.show()


# In[ ]:


#Bar Plot
ir1.plot(kind='bar',figsize=(6,6))
ir1.plot(kind='bar',stacked=True)


# In[ ]:


#Scatter plot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(ir1)
plt.show()


# In[ ]:


from sklearn import model_selection


# In[ ]:


#Splitout Validation Dataset
array=ir1.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.20
seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


#model building and spot check Algorithms
seed=7
scoring='accuracy'
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
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_result=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    msg="%s:%f (%f)" %(name,cv_result.mean(),cv_result.std())
    print(msg)


# In[ ]:


#Algorithm Comparision
fig=plt.figure(figsize=(10,6))
fig.suptitle('Algorithm Comparision')
ax=fig.add_subplot(111)
plt.boxplot(result)
ax.set_xticklabels(names)
plt.show()


# In[ ]:


#predictions on validation dataset
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
prediction=knn.predict(X_validation)


# In[ ]:


print(accuracy_score(Y_validation,prediction))


# In[ ]:


print(confusion_matrix(Y_validation, prediction))


# In[ ]:


print(classification_report(Y_validation,prediction))

