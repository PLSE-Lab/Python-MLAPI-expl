#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset_datasetpath='../input/titanic/train.csv'
dataset=pd.read_csv(dataset_datasetpath)

dataset_test_filepath='../input/titanic/test.csv'
dataset_test=pd.read_csv(dataset_test_filepath)


# In[ ]:


print(dataset.head())
dataset_test.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


mean=dataset['Age'].mean()
dataset['Age']=dataset['Age'].fillna(mean)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset['Embarked']


# In[ ]:


import seaborn as sns
plt.figure(figsize=(30,10))
sns.barplot(x=dataset.index,y=dataset['Age'])


# In[ ]:


# cleaning the cabin
dataset['Cabin'].unique().tolist()


# In[ ]:


cabin=dataset.groupby('Cabin')['Age']
print(cabin.mean())


# In[ ]:


dataset['Cabin'].unique().tolist()


# In[ ]:


train_dataset=dataset


# In[ ]:


train_dataset['Cabin'].unique()


# In[ ]:


train_dataset['Cabin'].fillna('U')


# In[ ]:


train_dataset['Cabin']=train_dataset['Cabin'].fillna('U')


# In[ ]:


train_dataset.isnull().sum()


# In[ ]:


# cleaning the test dataset
dataset_test.isnull().sum()


# In[ ]:


mean=dataset['Age'].mean()
print(mean)
dataset_test['Age']=dataset_test['Age'].fillna(mean)


# In[ ]:


dataset_test['Age']


# In[ ]:


dataset_test.head()


# In[ ]:


mean_fare=dataset_test['Fare'].mean()
mean_fare
dataset_test['Fare']=dataset_test['Fare'].fillna(mean_fare)


# In[ ]:


dataset_test.isnull().sum()


# In[ ]:


# take care of Cabin

dataset_test['Cabin']=dataset_test['Cabin'].fillna('U')


# In[ ]:


dataset_test['Cabin'].unique()


# In[ ]:


train_dataset.head()


# In[ ]:


dataset_test.head()


# In[ ]:


X_train=train_dataset.iloc[:,[2,5,6,7,9,11]]
X_train.head()


# In[ ]:


Y_train=train_dataset.iloc[:,1]
Y_train


# In[ ]:


X_test=dataset_test.iloc[:,[1,4,5,6,8,10]]
X_test


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X_train.iloc[:,5]=labelencoder_X.fit_transform(X_train.iloc[:,5])
X_train


# In[ ]:


labelencoder_Xtest=LabelEncoder()
X_test.iloc[:,5]=labelencoder_X.fit_transform(X_test.iloc[:,5])
X_test


# In[ ]:


dummy=pd.get_dummies(X_train["Embarked"])
dummy


# In[ ]:


X_train=X_train.merge(dummy,left_index=True,right_index=True)
X_train


# In[ ]:


X_train_main=X_train.iloc[:,[0,1,2,3,4,7,8]]
X_train_main


# In[ ]:


dummy=pd.get_dummies(X_test["Embarked"])
dummy


# In[ ]:


X_test=X_test.merge(dummy,left_index=True,right_index=True)
X_test


# In[ ]:


X_test_main=X_test.iloc[:,[0,1,2,3,4,7,8]]
X_test_main


# In[ ]:


X_train_main


# In[ ]:


Y_train


# In[ ]:


X_test_main


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train_main,Y_train)


# In[ ]:


y_pred=classifier.predict(X_test_main)


# In[ ]:


y_pred


# In[ ]:



# TO CHECK THE ACCURACY WE HAVE TO DO THIS STEP
from sklearn.model_selection import train_test_split
X_train,Y_train,X_test,Y_tes=train_test_split(X_train_main,Y_train,test_size=0.2,random_state=0)


# In[ ]:


X_train.shape


# In[ ]:


Y_train.shape


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,X_test)


# In[ ]:


y_pred1=classifier.predict(Y_train)


# In[ ]:


y_pred1


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_tes,y_pred1)


# In[ ]:


cm


# In[ ]:


accuracy=(94+33)/(94+33+16+36)
accuracy


# In[ ]:




