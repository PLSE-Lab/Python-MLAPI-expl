#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.graph_objs as go
import plotly.offline as py


# In[ ]:


dietanic_data=pd.read_csv("../input/titanic/train.csv")


# In[ ]:


dietanic_data1=pd.read_csv("../input/titanic/test.csv")


# In[ ]:


dietanic_data.head()


# In[ ]:


dietanic_data.info()


# In[ ]:


dietanic_data.isnull().sum()


# In[ ]:


dietanic_data.columns


# In[ ]:


dietanic_data["Survived"].describe()


# In[ ]:





# In[ ]:


dietanic_data['Survived'].value_counts().plot.bar()


# In[ ]:


sns.countplot('Survived',data=dietanic_data)


# In[ ]:


dietanic_data['Survived'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)


# In[ ]:


plt.figure(figsize=(3,5))
sns.barplot(y="Survived", x="Sex",hue="Sex",data=dietanic_data)


# In[ ]:


dietanic_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()


# In[ ]:


pd.crosstab(dietanic_data.Sex,dietanic_data.Survived,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


pd.crosstab(dietanic_data.Pclass,dietanic_data.Survived,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


f,axes=plt.subplots(1,2,figsize=(18,8))
dietanic_data['Pclass'].value_counts().plot.bar(ax=axes[0])
axes[0].set_title("Passengers")
sns.countplot(x='Pclass',hue='Survived',data=dietanic_data)


# In[ ]:


pd.crosstab([dietanic_data.Pclass,dietanic_data.Sex],dietanic_data.Survived,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=dietanic_data)
plt.show()


# In[ ]:


sns.barplot('Sex','Survived',hue='Pclass',data=dietanic_data)


# # age

# In[ ]:


dietanic_data.Age.describe()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass",'Age',hue="Survived",data=dietanic_data,ax=ax[0])
sns.violinplot("Sex",'Age',hue="Survived",data=dietanic_data,ax=ax[1])
plt.show()


# In[ ]:


dietanic_data.isna().sum()


# In[ ]:


dietanic_data['Initial']=0
for i in dietanic_data:
    dietanic_data['Initial']=dietanic_data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations


# In[ ]:


dietanic_data.head()


# In[ ]:


pd.crosstab(dietanic_data.Initial,dietanic_data.Sex)


# In[ ]:


dietanic_data["Initial"].replace(['Countess','Dr','Don','Capt','Col','Jonkheer','Lady','Major','Mlle','Mme','Ms','Rev','Sir'],['Mrs','Mr','Mr','Mr','Other','Other','Miss','Other','Mrs','Mrs','Miss','Other','Mr'],inplace=True)


# In[ ]:


pd.crosstab(dietanic_data.Initial,dietanic_data.Sex)


# In[ ]:


dietanic_data.groupby('Initial')["Age"].mean()


# In[ ]:


## Assigning the NaN Values with the Ceil values of the mean ages
dietanic_data.loc[(dietanic_data.Age.isnull())&(dietanic_data.Initial=='Mr'),'Age']=33
dietanic_data.loc[(dietanic_data.Age.isnull())&(dietanic_data.Initial=='Mrs'),'Age']=36
dietanic_data.loc[(dietanic_data.Age.isnull())&(dietanic_data.Initial=='Master'),'Age']=5
dietanic_data.loc[(dietanic_data.Age.isnull())&(dietanic_data.Initial=='Miss'),'Age']=22
dietanic_data.loc[(dietanic_data.Age.isnull())&(dietanic_data.Initial=='Other'),'Age']=46


# In[ ]:


dietanic_data.isna().sum()


# In[ ]:


dietanic_data['Embarked'].fillna('S',inplace=True)


# In[ ]:


dietanic_data=dietanic_data.drop(["Cabin"],axis=1)


# In[ ]:


dietanic_data.isna().sum()


# # heatmap

# In[ ]:


sns.heatmap(dietanic_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #dietanic_data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# # Prediction time

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# In[ ]:


train_y=dietanic_data[["Survived"]]


# In[ ]:


train_y.head()


# In[ ]:


train_x=dietanic_data


# In[ ]:


train_x=train_x.drop(["Initial","Name","Survived","PassengerId","Ticket"],axis=1)


# In[ ]:


train_x.isna().sum()


# In[ ]:


train_x.info()


# ### dummy variabls for categorical variables

# In[ ]:


train_x = pd.get_dummies(train_x)


# In[ ]:


train_x.head()


# In[ ]:


train_x.shape


# # Splitting just for Practice

# In[ ]:


X_train, X_test, y_train, y_test =train_test_split(train_x,
                    train_y,test_size=0.3,random_state=231)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


#TEST DATA


# In[ ]:


dietanic_data1=dietanic_data1.drop(["Name","PassengerId","Ticket","Cabin"],axis=1)


# In[ ]:


dietanic_data1.isna().sum()


# In[ ]:


dietanic_data1.Age.fillna(dietanic_data1.Age.mean(),
                inplace=True)


# In[ ]:


dietanic_data1.Fare.fillna(dietanic_data1.Fare.mean(),
                inplace=True)


# In[ ]:


dietanic_data1.isna().sum()


# In[ ]:


dietanic_data1 = pd.get_dummies(dietanic_data1)


# ## Radial Support Vector Machines(rbf-SVM

# In[ ]:


model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_x,train_y)
prediction1=model.predict(dietanic_data1)


# In[ ]:


prediction1


# ## Linear Support Vector Machine(linear-SVM)

# In[ ]:


model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_x,train_y)
prediction2=model.predict(dietanic_data1)


# ## Logistic Regression

# In[ ]:


model = LogisticRegression()
model.fit(train_x,train_y)
prediction3=model.predict(dietanic_data1)


# ## Decision tree

# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_x,train_y)
prediction4=model.predict(dietanic_data1)


# ## K-Nearest Neighbours(KNN)

# In[ ]:


model=KNeighborsClassifier() 
model.fit(train_x,train_y)
prediction5=model.predict(dietanic_data1)


# ## Gaussian Naive Bayes

# In[ ]:


model=GaussianNB()
model.fit(train_x,train_y)
prediction6=model.predict(dietanic_data1)


# ## Random Forests

# In[ ]:


model=RandomForestClassifier(n_estimators=100)
model.fit(train_x,train_y)
prediction7=model.predict(dietanic_data1)


# In[ ]:


final_pred=pd.DataFrame(prediction6)


# In[ ]:


sub_df=pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


datasets=pd.concat([sub_df['PassengerId'],final_pred],axis=1)
datasets.columns=['PassengerId','Survived']
datasets.to_csv('sample_submission.csv',index=False)


# In[ ]:




