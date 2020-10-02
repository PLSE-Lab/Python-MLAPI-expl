#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# 

# In[ ]:


data=pd.read_csv('../input/cs-training.csv')


# In[ ]:


data.head()


# In[ ]:


print(data.isnull().sum())


# In[ ]:


data.describe()


# In[ ]:


data['MonthlyIncome'].fillna(data['MonthlyIncome'].mean(),inplace=True)


# In[ ]:


data['NumberOfDependents'].fillna(data['NumberOfDependents'].mode()[0], inplace=True)


# In[ ]:


print(data.isnull().sum())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


cor=data.corr()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,annot=True,ax=ax)


# In[ ]:


attributes=['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome']
sol=['SeriousDlqin2yrs']
X=data[attributes]
y=data[sol]


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier #Since the Dataset is Imbalanced, we go for boosting model.
model = XGBClassifier(tree_method = 'gpu_exact')
model.fit(X,y.values.ravel())
y_pred = model.predict(X)
print("The Accuracy score is : ",accuracy_score(y,y_pred)*100,"%")
print(confusion_matrix(y,y_pred))


# In[ ]:


from sklearn.model_selection import KFold,cross_val_score
kf = KFold(n_splits=5, random_state=None) 
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
print(np.mean(cross_val_score(model, X, y.values.ravel(), cv=10))*100)


# In[ ]:


data_test=pd.read_csv('../input/cs-test.csv')


# In[ ]:


print(data_test.isnull().sum())


# In[ ]:


data_test['MonthlyIncome'].fillna(data_test['MonthlyIncome'].mean(),inplace=True)


# In[ ]:


xtest=data_test[attributes]


# In[ ]:


xtest.head()


# In[ ]:


ytest=model.predict_proba(xtest)


# In[ ]:


print(ytest)


# In[ ]:


df=pd.DataFrame(ytest,columns=['Id','Probability'])


# In[ ]:


df.head()


# In[ ]:


ind=data['Unnamed: 0']


# In[ ]:


df['Id']=ind


# In[ ]:


df.head()


# In[ ]:


export_csv = df.to_csv('export_dataframe.csv',index = None,header=True)

