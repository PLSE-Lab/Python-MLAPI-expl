#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


cancer = pd.read_csv('../input/data.csv')
cancer.head(20)


# In[ ]:


cancer.keys()


# In[ ]:


cancer.describe()


# In[ ]:


cancer.info()


# In[ ]:


diagnosis_target=pd.get_dummies(cancer['diagnosis'],drop_first=True) # convert Diagnosis (M = malignant, B = benign) ==> M=1 , B=0
diagnosis_target


# In[ ]:


cancer=pd.concat([cancer,diagnosis_target],axis=1)  # axis=1 ==>rows   , axis=0 ==>cols
cancer.head()


# In[ ]:


sns.pairplot(cancer,hue='M',vars=['radius_mean','texture_mean','area_mean','perimeter_mean','smoothness_mean'])


# In[ ]:


print('benign = ',len( cancer[cancer['diagnosis']=='B']))
print('malignant = ',len( cancer[cancer['diagnosis']=='M']))


# In[ ]:


sns.countplot(cancer['M'],label='Count')


# In[ ]:


sns.scatterplot(x = 'area_mean' , y = 'smoothness_mean' ,hue = 'M' , data = cancer )


# In[ ]:


sns.scatterplot(x = 'perimeter_mean' , y = 'texture_mean' ,hue = 'M' , data = cancer )


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(cancer.corr(),annot=True)


# In[ ]:


X=cancer.drop(['M'],axis=1)
X=X.drop(['diagnosis'],axis=1)
X=X.drop(['Unnamed: 32'],axis=1)
X


# In[ ]:


y=cancer['M']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

svc_model = SVC()
svc_model.fit(X_train,y_train)


# In[ ]:


y_predict = svc_model.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_predict))


# # Impoving the Model

# In[ ]:


min_train=X_train.min()
min_train


# In[ ]:


max_train=X_train.max()
max_train


# In[ ]:


range_train=(X_train - min_train).max()
range_train


# In[ ]:


X_train_scaled =(X_train - min_train) / range_train
X_train_scaled


# In[ ]:


sns.scatterplot(x = X_train['area_mean'], y = X_train['smoothness_mean'] ,hue = y_train)


# In[ ]:


sns.scatterplot(x = X_train_scaled['area_mean'], y = X_train_scaled['smoothness_mean'] ,hue = y_train)


# In[ ]:


min_test = X_test.min()
max_test = X_test.max()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test) / range_test


# In[ ]:


sns.scatterplot(x = X_test['area_mean'], y = X_test['smoothness_mean'] ,hue = y_test)


# In[ ]:


sns.scatterplot(x = X_test_scaled['area_mean'], y = X_test_scaled['smoothness_mean'] ,hue = y_test)


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled,y_train)


# In[ ]:


Y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot = True)


# In[ ]:


print(classification_report(y_test,y_predict))


# In[ ]:


param_grid = { 'C':[0.1,1,10,100] , 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf'] }


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid, refit = True , verbose=4)
grid.fit(X_train_scaled,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_predict = grid.predict(X_test_scaled)
dm = confusion_matrix(y_test,grid_predict)
sns.heatmap(cm, annot = True)


# In[ ]:


print(classification_report(y_test,grid_predict))

