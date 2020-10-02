#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data=pd.read_csv('../input/breastcancer.csv')


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# Checking null values

# In[ ]:


data.isnull().sum()


# This data sheet contains 569 rows and 32 columns including target column. 

# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.groupby(['diagnosis']).size()


# Total 357 benign and 212 malignant cancer. 

# Visualizing the data

# In[ ]:


sns.countplot(x='diagnosis',data=data)


# In[ ]:


sns.pairplot(data=data,hue='diagnosis',vars=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean'])


# In[ ]:


sns.scatterplot(data=data,x='area_mean',y='smoothness_mean',hue='diagnosis')


# Checking Correlation between variables.

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True,cmap='Blues')
plt.show()


# High correlation between perimiter_mean & area_mean and radius_mean & perimeter_mean.

# Model training

# In[ ]:


X=data.drop(['diagnosis','id'],axis=1)
y=data['diagnosis']


# In[ ]:


X


# In[ ]:


y


# Splitting the data
# 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# Fitting the model

# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

lr=LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
y_predl=lr.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy of training set',accuracy_score(y_train,lr.predict(X_train)))
print('Accuracy of testing set ',accuracy_score(y_test,y_predl))


# Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_predd=clf.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy of training set',accuracy_score(y_train,clf.predict(X_train)))
print('Accuracy of testing set ',accuracy_score(y_test,y_predd))


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_predr=rf.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy of training set',accuracy_score(y_train,rf.predict(X_train)))
print('Accuracy of testing set ',accuracy_score(y_test,y_predr))


# Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train,y_train)
y_preds=sv.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy of training set',accuracy_score(y_train,sv.predict(X_train)))
print('Accuracy of testing set ',accuracy_score(y_test,y_preds))


# KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_predk=knn.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy of training set',accuracy_score(y_train,knn.predict(X_train)))
print('Accuracy of testing set ',accuracy_score(y_test,y_predk))


# Evaluating the model with Support Vector

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_preds)
cm


# In[ ]:


sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_preds))


# Improving the model with SVM

# In[ ]:


min_train=X_train.min()
min_train


# In[ ]:


new_train=(X_train-min_train).max()
X_train_scaled=(X_train-min_train)/new_train
X_train_scaled


# Scatter plot before scaling

# In[ ]:


sns.scatterplot(x=X_train['area_mean'],y=X_train['smoothness_mean'],hue=y_train)


# Scatter plot after scaling

# In[ ]:


sns.scatterplot(x=X_train_scaled['area_mean'],y=X_train_scaled['smoothness_mean'],hue=y_train)


# Now scale the test data also.

# In[ ]:


min_test=X_test.min()
new_test=(X_test-min_test).max()
X_test_scaled=(X_test-min_test)/new_test
X_test_scaled


# Again fit the model and evaluate the result

# In[ ]:


sv=SVC()
sv.fit(X_train_scaled,y_train)
y_preds=sv.predict(X_test_scaled)
from sklearn.metrics import accuracy_score
print('Accuracy of training set',accuracy_score(y_train,sv.predict(X_train_scaled)))
print('Accuracy of testing set ',accuracy_score(y_test,y_preds))


# In[ ]:


cm=confusion_matrix(y_test,y_preds)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_preds))


# Again we can try some more improvements using Gridsearch.Lets do it...

# In[ ]:


param_grid={'C':[0.1,1,10,50,100],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[ ]:


grid.fit(X_train_scaled,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_pred=grid.predict(X_test_scaled)


# In[ ]:


cm=confusion_matrix(y_test,grid_pred)
cm


# In[ ]:


sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,grid_pred))

