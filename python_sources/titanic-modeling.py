#!/usr/bin/env python
# coding: utf-8

# # import the Libraries 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


# # import the data set

# In[ ]:


data = pd.read_csv('../input/titanic/train.csv',index_col = "PassengerId")
test = pd.read_csv('../input/titanic/test.csv',index_col = "PassengerId")

#test_y = pd.read_csv('../dataset/titanic/gender_submission.csv',index_col = "PassengerId")


# In[ ]:


indexs=  test.index


# In[ ]:


data.sample(2)


# In[ ]:


data.info()


# In[ ]:


data.shape


# # specify the features and dependent variable

# In[ ]:


X = data.iloc[:,1:]
y = data.iloc[:,0]


# In[ ]:


X.columns


# In[ ]:


X['Ticket'].mode


# In[ ]:


y


# ## dealing with missing values

# In[ ]:


X.sample()


# In[ ]:


X =X.drop(columns =['Name'])


# In[ ]:


X.info()


# In[ ]:


from sklearn.impute import SimpleImputer
imputer_no = SimpleImputer(missing_values= np.nan ,strategy = 'mean')
imputer_no.fit(X[['Pclass','Age','SibSp','Fare','Parch']])
X[['Pclass','Age','SibSp','Fare','Parch']] = imputer_no.transform(X[['Pclass','Age','SibSp','Fare','Parch']])


# In[ ]:


imputer_cat = SimpleImputer(missing_values= np.nan ,strategy = 'most_frequent')
imputer_cat.fit(X[['Sex','Cabin','Embarked','Ticket']])
X[['Sex','Cabin','Embarked','Ticket']]=imputer_cat.transform(X[['Sex','Cabin','Embarked','Ticket']])


# In[ ]:


X.info()


# # convert String to binary

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 


# In[ ]:


ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(handle_unknown='ignore'),[1,5,7,8])],remainder = 'passthrough')
X=  ct.fit_transform(X)


# ## Split our data 

# In[ ]:


from sklearn.model_selection import train_test_split 
train_X,test_X,train_y,test_y = train_test_split(X,y)


# # train the model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
for i in range(10,300,10):
    classifier = RandomForestClassifier(n_estimators= i,criterion='gini' )
    classifier.fit(train_X, train_y)
    y_predict = classifier.predict(test_X)
    print('for {} estimators and {}'.format({i},{accuracy_score(y_true=test_y,y_pred=y_predict)}))


# In[ ]:


test.columns


# In[ ]:


test.info()


# In[ ]:


test =test.drop(columns =['Name'])


# In[ ]:



imputer_no.fit(test[['Pclass','Age','SibSp','Fare','Parch']])
test[['Pclass','Age','SibSp','Fare','Parch']] = imputer_no.transform(test[['Pclass','Age','SibSp','Fare','Parch']])
imputer_cat.fit(test[['Sex','Cabin','Embarked','Ticket']])
test[['Sex','Cabin','Embarked','Ticket']]=imputer_cat.transform(test[['Sex','Cabin','Embarked','Ticket']])
test=  ct.transform(test)


# In[ ]:


test


# In[ ]:


classifier = RandomForestClassifier(n_estimators= 150,criterion='gini')
classifier.fit(X, y)
y_predict = classifier.predict(test)
pd.DataFrame(y_predict,index=indexs,columns=['Survived'] ).to_csv('output.csv')


# In[ ]:


y_predict


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Encoding Categorical Data

# In[ ]:


#def unique_values(my_col):
   # return my_col.nunique()


# In[ ]:


#train_X.apply(unique_values,axis = 0)


# In[ ]:


#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder 


# In[ ]:


#ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[2])],remainder = 'passthrough')
#ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[9])],remainder = 'passthrough')
#train_X = ct.fit_transform(train_X)


# In[ ]:


#pd.DataFrame(train_X)


# In[ ]:


#train_X.describe()

