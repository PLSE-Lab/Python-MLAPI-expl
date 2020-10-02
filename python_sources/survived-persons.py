#!/usr/bin/env python
# coding: utf-8

# # 1) Acquire Data

# ## Import pandas

# In[173]:


import pandas as pd


# In[174]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


ids = test_df['PassengerId']


# # 02) Analyze by describing data

# In[175]:


print(train_df.columns.values)


# # 03) preview the data

# In[176]:


train_df.head()


# In[177]:


train_df.tail()


# In[178]:


train_df.info()
print('_'*40)
test_df.info()


# In[179]:


train_df.describe()


# In[180]:


train_df=train_df.drop(columns=['Name','Cabin','Ticket','PassengerId'])

test_df=test_df.drop(columns=['Name','Cabin','Ticket','PassengerId'])

train_df.shape


# In[181]:


survived = train_df['Survived']

features = train_df.drop('Survived',axis=1)


# In[182]:


from sklearn.preprocessing import Imputer
my_imputer = Imputer('NaN','median')
features['Age'] = my_imputer.fit_transform(features[['Age']])
test_df['Age'] = my_imputer.fit_transform(test_df[['Age']])

features['Fare'] = my_imputer.fit_transform(features[['Fare']])
test_df['Fare'] = my_imputer.fit_transform(test_df[['Fare']])


# In[183]:


features=pd.get_dummies(features)

test_df=pd.get_dummies(test_df)


# In[184]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(survived)

survived = le.transform(survived)


# In[185]:


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    survived,random_state=0)


# In[186]:


def prediction(Model,X_train,y_train,X_test,y_test) :
    
    clf=Model()
    
    clf.fit(X_train,y_train)
    
    print(clf.score(X_test,y_test))
    
    return clf


# In[187]:


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

clf_A = prediction(RandomForestClassifier,X_train,y_train,X_test,y_test)

clf_B = prediction(MLPClassifier,X_train,y_train,X_test,y_test)

clf_C = prediction(AdaBoostClassifier,X_train,y_train,X_test,y_test)


# In[ ]:


predictions = clf_C.predict(test_df)


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)


# In[ ]:





# In[ ]:




