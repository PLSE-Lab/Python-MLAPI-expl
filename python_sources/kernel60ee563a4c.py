#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


from pandas import DataFrame


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


import numpy as np


# In[ ]:


from matplotlib.legend_handler import HandlerLine2D


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.metrics import roc_curve, auc


# In[ ]:


home_file=pd.read_csv('../input/breastcancer-dataset/data.csv')


# In[ ]:


home_file.describe()


# In[ ]:


home_file['diagnosis'].value_counts()


# In[ ]:


home_file.shape


# In[ ]:


home_file.keys()


# In[ ]:


X=home_file.drop(['diagnosis', 'Unnamed: 32'], axis=1)


# In[ ]:


y=home_file.diagnosis.map(dict(M=1,B=0))


# In[ ]:


home_file.info()


# In[ ]:


y.dtype


# In[ ]:


print(y.tail())


# In[ ]:


X.head()


# In[ ]:


X.head()


# In[ ]:


sns.pairplot(home_file, hue='diagnosis', vars= ['radius_mean', 'texture_mean','area_mean','perimeter_mean','smoothness_mean'])


# In[ ]:


sns.scatterplot(x='area_mean', y='compactness_mean', hue='diagnosis', data=home_file)


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(home_file.corr(), annot=True)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=0)


# In[ ]:


X_train.head()


# In[ ]:


y_test[:5]


# In[ ]:


my_model=RandomForestClassifier(n_estimators=100,random_state=0)


# In[ ]:


my_model.fit(X_train, y_train)


# In[ ]:


y_prediction=my_model.predict(X_test)


# In[ ]:


np.mean(y_prediction==y_test)


# In[ ]:


my_model.score(X_test,y_test)


# In[ ]:


my_model.score(X_train,y_train)


# In[ ]:


#to check the auc for the random_forest without tunning
false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_prediction)


# In[ ]:


roc_auc=auc(false_positive_rate, true_positive_rate)


# In[ ]:


roc_auc


# In[ ]:


#to checkmate overfitting, we try to see the best number of estimators or finetune our model
n_estimators=[1,2,4,8,16,32,64,100,200]

train_results=[]
test_results=[]
for estimator in n_estimators:
    my_model=RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    my_model.fit(X_train, y_train)
    
    y_prediction1=my_model.predict(X_train) #we are monitoring the curve for predicting X_train values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_train, y_prediction1)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    train_results.append(roc_auc)
    
    
    y_prediction2=my_model.predict(X_test) # NOW, we are monitoring the curve for predicting X_test values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_prediction2)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    test_results.append(roc_auc)
    
    
    


# In[ ]:


test_results


# In[ ]:


line1,=plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2,=plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


# In[ ]:


#lets see the range we need to consider in choosing our max_depth
max_depths=np.linspace(1,32,32, endpoint=True)

train_results=[]
test_results=[]
for max_depth in max_depths:
    my_model=RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    my_model.fit(X_train, y_train)
    
    y_prediction1=my_model.predict(X_train) #we are monitoring the curve for predicting X_train values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_train, y_prediction1)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    train_results.append(roc_auc)
    
    
    y_prediction2=my_model.predict(X_test) # NOW, we are monitoring the curve for predicting X_test values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_prediction2)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    test_results.append(roc_auc)
    
    
    


# In[ ]:


line1,=plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2,=plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


# In[ ]:


my_model=RandomForestClassifier(n_estimators=100,max_depth=3, n_jobs=-1)
my_model.fit(X_train, y_train)


# In[ ]:


y_prediction2=my_model.predict(X_test)


# In[ ]:


my_model.score(X_test,y_test)


# In[ ]:


my_model.score(X_train,y_train)


# In[ ]:


y_new=my_model.predict(X)


# In[ ]:


new_table=pd.DataFrame({'id':home_file['id'],'diagnosis':y_new} )


# In[ ]:


new_table.head()

