#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dataset=pd.read_csv('/kaggle/input/titanic/train.csv')
dataset.head()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.shape


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# In[ ]:


dataset=dataset.drop('Ticket',axis=1)


# In[ ]:


dataset['Cabin']=dataset['Cabin'].fillna('U')
dataset['Cabin']=dataset.Cabin.apply(lambda c:c[0])
dataset.head()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=dataset,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=dataset,palette='RdBu_r')


# From above plot we can conclude that females have survived more than the males 
# 

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=dataset,palette='rainbow')


# Passenger class is important feature as we can see that lower class 
# passengers have more fatalities than higher class passengers

# In[ ]:


sns.distplot(dataset['Age'].dropna(),kde=False,color='darkred',bins=30)


# Most of the passengers are young and ranges from 20-30 years of age

# Now we wil use heatmap to account for the missing values

# In[ ]:


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Most of the missing values are there in cabin section so we can drop it for now
# but we have some missing values in age section so we have to replace it with proper 
# values

# In[ ]:


dataset.head()


# Name can be handy in deciding the proper values to be filled in the age section 
# so i decided to extract the salutation from the name and compute their median age

# In[ ]:


dataset['Title']=dataset.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip())


# In[ ]:


dataset['Title'].value_counts()


# In[ ]:


group=dataset.groupby(['Sex','Pclass','Title'])
group.Age.median()


# In[ ]:


#filling the age with the median values according to the salutation
dataset.Age=group.Age.apply(lambda x:x.fillna(x.median()))


# In[ ]:


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#remove passenger id as it is of no use
dataset.drop('PassengerId',inplace=True,axis=1)


# In[ ]:


dataset.head()


# In[ ]:


# here i will fill the missing values in embarked with the most frequent values
# in it.  

emb_most=dataset.Embarked.value_counts().index[0]
dataset.Embarked=dataset.Embarked.fillna(emb_most)
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#removing the cabin section
dataset.drop('Cabin',inplace=True,axis=1)

# encoding categorical variable and removing one of the dummies variable to avoid
# dummies variable trap.
embark=pd.get_dummies(dataset['Embarked'],drop_first=True)
sex=pd.get_dummies(dataset['Sex'],drop_first=True)

# now we will concatenate the dummies variable with the original dataset.
dataset.drop(['Sex','Embarked'],axis=1,inplace=True)
dataset=pd.concat([dataset,sex,embark],axis=1)

#Similarly we will encode pclass feature
pclass=pd.get_dummies(dataset['Pclass'],drop_first=True)
dataset.drop(['Pclass'],axis=1,inplace=True)
dataset=pd.concat([dataset,pclass],axis=1)

#renaming the dummies variable for passenger class for simplicity
dataset=dataset.rename(columns={2:'Pclass2',3:'Pclass3'})


# In[ ]:


dataset.head()


# In[ ]:


#Now we will analyze the survival rate of passengers according to their Family size
# including siblings(SibSp) and Parent-Child(Parch)
dataset.groupby(['SibSp','Parch']).Survived.median()


# For simplicty i m going to define a new feature called Family Size to analyze 
# it with ease

# In[ ]:


# Creating new feature of Family size
dataset['Family_Size']=dataset.SibSp+dataset.Parch+1
dataset[['Family_Size','Survived']].groupby('Family_Size').Survived.mean().sort_values(ascending=False)


# We can see from above that the one with large family size has less survival rate

# In[ ]:


# Now with the help of heatmap we will see correlations between the features
plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(),annot=True)


# From the above correlation matrix we can see that  family size have high correlation with the sibling and Parch which is obvious so we will drop it.

# In[ ]:


dataset.drop(['Title','Name','SibSp','Parch'],axis=1,inplace=True)


# In[ ]:


dataset.head()


# # Now we will similarly handle the test data set

# In[ ]:


testdata=pd.read_csv('/kaggle/input/titanic/test.csv')
testdata.head()


# In[ ]:


testdata.isnull().sum()


# In[ ]:


testdata.shape


# In[ ]:


testdata=testdata.drop('Cabin',axis=1)


# In[ ]:


group=testdata.groupby(['Pclass'])
group.Fare.median()


# In[ ]:


testdata.Fare=group.Fare.apply(lambda x:x.fillna(x.median()))


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(testdata.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


testdata['Title']=testdata.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip())
group1=testdata.groupby(['Sex','Pclass','Title'])
group1.Age.median()
testdata.Age=group1.Age.apply(lambda x:x.fillna(x.median()))
testdata['Title'].value_counts()


# In[ ]:


testdata[testdata['Title'].isnull()]


# In[ ]:


grp=testdata.groupby(['Pclass','Sex'])
grp.Age.median()
testdata.Age=grp.Age.apply(lambda x:x.fillna(x.median()))


# In[ ]:


testdata.isnull().sum()


# In[ ]:


embark=pd.get_dummies(testdata['Embarked'],drop_first=True)
sex=pd.get_dummies(testdata['Sex'],drop_first=True)
testdata.drop(['Sex','Embarked'],axis=1,inplace=True)
testdata=pd.concat([testdata,sex,embark],axis=1)
pclass=pd.get_dummies(testdata['Pclass'],drop_first=True)
testdata.drop(['Pclass'],axis=1,inplace=True)
testdata=pd.concat([testdata,pclass],axis=1)
testdata=testdata.rename(columns={2:'Pclass2',3:'Pclass3'})


# In[ ]:


testdata['Family Size']=testdata.SibSp+testdata.Parch+1
testdata.head()


# In[ ]:


testdata.drop(['Title','Name','Ticket','SibSp','Parch'],axis=1,inplace=True)
testdata.head()


# # Now we will train our Machine learning model on training dataset and fit it over test data set to predict the survival using Random Forest Classifier.

# In[ ]:


from sklearn.model_selection import train_test_split as t
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
X=dataset.drop('Survived',axis=1)
y=dataset['Survived']


# In[ ]:


#first we will divide our train dataset to further train and test dataset to check its accuracy
X_train,X_test,y_train,y_test=t(X,y,test_size=.2,random_state=0)


# In[ ]:


#defining parameters of Random forrest classifier for hyperParameter tuning.
forrest_params = { 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],     
    
                 }


# In[ ]:


forrest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forrest,param_grid=forrest_params, cv=5) 
forest_cv.fit(X_train, y_train)


# In[ ]:


y_pred = forest_cv.predict(X_test)


# In[ ]:


forest_cv.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# Now we will fit it over all train dataset

# In[ ]:


forest_cv.fit(X,y)
y_pred=forest_cv.predict(X)
forest_cv.score(X,y)


# In[ ]:


#now at last we will predict on test data set
y_pred=forest_cv.predict(testdata.drop('PassengerId',axis=1))


# In[ ]:


data=pd.DataFrame({'PassengerId':testdata['PassengerId'],'Survived':y_pred})


# In[ ]:


data.to_csv('submission.csv',index=False)


# I got a accuracy of .81818 on a test dataset and i am trying to improve it by doing more exploratory data analysis.
# Please be free to share some suggestions if possible.
# Thank you!

# In[ ]:




