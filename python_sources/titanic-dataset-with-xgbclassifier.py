#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


Sex=pd.get_dummies(data['Sex'],drop_first=True)
Sex.head()


# In[ ]:


Embarked=pd.get_dummies(data['Embarked'],drop_first=True)
Embarked.head(5)


# In[ ]:


Pclass=pd.get_dummies(data['Pclass'],drop_first=True)
Pclass.head(5)


# In[ ]:


data=pd.concat([data,Sex,Embarked,Pclass],axis=1)


# In[ ]:


data.head(5)


# In[ ]:


data.drop(['Sex','Embarked','Name','Cabin','Ticket'],axis=1,inplace=True)


# In[ ]:


data.head(5)


# In[ ]:


f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()


# In[ ]:


test.head()


# In[ ]:


Embarked=pd.get_dummies(test['Embarked'],drop_first=True)
Pclass=pd.get_dummies(test['Pclass'],drop_first=True)
Sex=pd.get_dummies(test['Sex'],drop_first=True)


# In[ ]:


test=pd.concat([test,Sex,Embarked,Pclass],axis=1)


# In[ ]:


test.drop(['Sex','Embarked','Name','Cabin','Ticket',],axis=1,inplace=True)
test.head(5)


# In[ ]:


data.head()


# In[ ]:


X_train = data.drop("Survived",axis=1)
print(X_train.head())
y_train = data["Survived"]
print(y_train.head())
X_test = test
print(X_test.head())
y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
y_test=y_test["Survived"]
print(y_test.head())


# In[ ]:


from xgboost import XGBClassifier
my_model = XGBClassifier()
my_model.fit(X_train,y_train, verbose=False)


# In[ ]:


predictions = my_model.predict(X_test)


# In[ ]:


predictions


# In[ ]:


from sklearn.metrics import accuracy_score

prediction = [round(value) for value in predictions]
accuracy = accuracy_score(y_test, prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


final_data = pd.DataFrame()
final_data['PassengerId'] = test['PassengerId']
final_data['Survived'] = predictions


# In[ ]:


final_data.to_csv(r'submission.csv',index=False)


# In[ ]:




