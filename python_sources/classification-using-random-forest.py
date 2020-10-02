#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install tabulate


# In[ ]:


import pandas as pd
import numpy as np
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


# In[ ]:


# Importing data in a dataframe
income_data = pd.read_csv('../input/adult-income-dataset/adult.csv',sep=r'\s*,\s*',engine = 'python')
income_data


# In[ ]:


# View the datatypes of each column
income_data.info()


# In[ ]:


# Finding the missing values column wise
income_data.isnull().sum()


# In[ ]:


#Checking unique and incorrect values in a column in a dataframe and replacing them
print(income_data.columns.tolist())
income_data['workclass'].unique()
income_data['workclass']= income_data['workclass'].replace({"?":"Unknown"})
income_data['workclass'].unique()


# In[ ]:


income_data['occupation'].unique()
income_data['occupation']= income_data['occupation'].replace({"?":"Unknown"})
income_data['occupation'].unique()


# In[ ]:


income_data['native-country'].unique()
income_data['native-country']= income_data['native-country'].replace({"?":"Unknown"})
income_data['native-country'].unique()


# In[ ]:


#Deleting redundant columns
del income_data['education']
del income_data['marital-status']
del income_data['fnlwgt']
income_data


# In[ ]:


# Plotting count by income
income_data['income'].value_counts().plot(kind = 'bar')


# In[ ]:


labels_sex = income_data['gender'].unique()
greater_than_50 = income_data[income_data.income == '>50K']
greater_than_50
less_than_50 = income_data[income_data.income == '<=50K']
greater_than_50['gender'].value_counts().plot(kind = 'bar')


# In[ ]:


less_than_50['gender'].value_counts().plot(kind = 'bar')


# In[ ]:



income_data.groupby(['relationship']).count().plot(kind='pie',y = 'income',legend=None)
plt.xlabel('Relationship')
plt.ylabel('Count')
#plt.legend( loc="upper left")
plt.show()


# In[ ]:


#Label encoding
label_encoder = preprocessing.LabelEncoder()
income_data['workclass']= label_encoder.fit_transform(income_data['workclass'])
income_data['occupation']= label_encoder.fit_transform(income_data['occupation'])
income_data['relationship']= label_encoder.fit_transform(income_data['relationship'])
income_data['race']= label_encoder.fit_transform(income_data['race'])
income_data['gender']= label_encoder.fit_transform(income_data['gender'])
income_data['native-country']= label_encoder.fit_transform(income_data['native-country'])
income_data['income']= label_encoder.fit_transform(income_data['income'])
income_data


# In[ ]:


#Heatmap
plt.figure(figsize=(14,10))
ax = sns.heatmap(income_data.corr(),annot=True,fmt='.2f')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


#Splitting train and test data
y = income_data.income
x_train,x_test,y_train,y_test = train_test_split(income_data,y,test_size = 0.2)
y_train = income_data.iloc[:,-1].values
x_train = income_data.iloc[:,:-1].values
y_test = income_data.iloc[:,-1].values
x_test = income_data.iloc[:,:-1].values


# In[ ]:


clf=RandomForestClassifier()
kf=KFold(n_splits=3)
max_features=np.array([1,2,3,4,5])
n_estimators=np.array([50,100,150,200])
min_samples_leaf=np.array([50,75,100,150])
param_grid=dict(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf)
grid=GridSearchCV(estimator=clf,param_grid=param_grid,cv=kf)
gres=grid.fit(x_train,y_train)
print("Best",gres.best_score_)
print("params",gres.best_params_)


# In[ ]:


rf = RandomForestClassifier(max_features= 2, min_samples_leaf= 50, n_estimators= 100)
rf.fit(x_train, y_train)


# In[ ]:


#Predicting for test set
pred=rf.predict(x_test)
pred


# In[ ]:


#Finding test accuracy
print("Accuracy: %f " % (100*accuracy_score(y_test, pred)))


# In[ ]:


#Confusion matrix
cm = confusion_matrix(y_test,pred)
cm


# In[ ]:


average_precision = average_precision_score(y_test, pred)
average_precision


# In[ ]:


recall_score(y_test, pred, average='macro')

