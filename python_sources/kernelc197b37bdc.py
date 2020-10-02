#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np 
import math 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[30]:


dp = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# age.describe() #decribe column for statiscial value
Y = dp['Survived'].copy()
X = dp[dp.columns.difference(['Survived'])].copy()


# In[78]:


dp.head()


# In[80]:


test.head()


# In[31]:


cols_with_missing = [col for col in dp.columns if dp[col].isnull().any()]
print(cols_with_missing)
cols_with_missing2 = [col for col in test.columns if test[col].isnull().any()]
print(cols_with_missing2)


# In[ ]:





# In[32]:


dp.groupby(dp['Survived']).count()


# In[33]:


X.describe()
# Age has missing data. We fix it by filling missing data with mean


# In[34]:


combined_data_set = [X,test]


# In[35]:


# female is 0; male is 1
for data_set in combined_data_set:
    data_set['Sex'] = data_set['Sex'].replace({'male':1,'female':0})


# In[36]:


for data_set in combined_data_set:
    #new entry for passengers embarked at S
    data_set['S'] = np.where(data_set['Embarked'] == 'S', 1,0)
    #new entry for passengers embarked at C
    data_set['C'] = np.where(data_set['Embarked'] == 'C', 1,0)
    #new entry for passengers embarked at Q
    data_set['Q'] = np.where(data_set['Embarked'] == 'Q', 1,0)
    #drop the remaining 'Embarked' column
    data_set.drop('Embarked',axis = 1,inplace = True)


# In[37]:


# Embarked: S is 0; C is 1; Q is 2
#for data_set in combined_data_set:
    #data_set['Embarked'] = data_set['Embarked'].replace({'S':0,'C':1,'Q':2})
    #data_set['Embarked'] = data_set['Embarked'].fillna(1)


# In[38]:


# using feature extraction to convert Name into a new feature called 'Titles'
for data_set in combined_data_set:
    data_set['Titles'] = data_set['Name'].str.extract(' ([A-Za-z]+)\.', expand= False)


# In[39]:


# we created a new feature called Titles: Mr: 0; Miss: 1; Mrs: 2; Others: 3
titles_mapping = {'Mr': 0, 'Miss': 1, 'Mrs':2}
for data_set in combined_data_set:
    data_set['Titles'] = data_set['Titles'].map(titles_mapping)
    data_set['Titles'] = data_set['Titles'].fillna(3)


# In[40]:


#X.groupby('Titles')['Age'].transform('max')
#k = pd.DataFrame({'k':[1,2,3,4,5,6], 'b': ['a','a','b','b','b','b'], 'g':[None,1,1, None,3,4]})
#k['g'].fillna(k.groupby('b')['g'].transform('median'), inplace = True)


# # fill the missing ages with median based on their title and normalize age
# for data_set in combined_data_set:
#     data_set['Age'].fillna(data_set.groupby('Titles')['Age'].transform('median'), inplace = True)
# 
#     data_set.loc[data_set['Age'] <= 7, 'Age'] = 0    
#     data_set.loc[data_set['Age'] > 7 & (data_set['Age'] <= 20),'Age'] = 1 
#     data_set.loc[data_set['Age'] > 20 & (data_set['Age'] <= 40),'Age'] = 2
#     data_set.loc[data_set['Age'] > 40 & (data_set['Age'] <= 60),'Age'] = 3
#     data_set.loc[data_set['Age'] > 60 & (data_set['Age'] <= 80),'Age'] = 4
#     data_set.loc[data_set['Age'] > 80 ,'Age'] = 5
#     
#     # separate ages into four different phrase
#     data_set['0_Age'] = np.where(data_set['Age'] == 0, 1,0)
#     data_set['1_Age'] = np.where(data_set['Age'] == 1, 1,0)
#     data_set['2_Age'] = np.where(data_set['Age'] == 2, 1,0)
#     data_set['3_Age'] = np.where(data_set['Age'] == 3, 1,0)
#     data_set['4_Age'] = np.where(data_set['Age'] == 4, 1,0)
#     data_set['5_Age'] = np.where(data_set['Age'] == 5, 1,0)
#     data_set.drop('Age',axis = 1,inplace = True)

# In[41]:


#creating spearate entries for all titles
for data_set in combined_data_set:
    #new entry for titles starts with Mr
    data_set['Mr'] = np.where(data_set['Titles'] == 0, 1,0)
    #new entry for titles starts with Miss
    data_set['Miss'] = np.where(data_set['Titles'] == 1, 1,0)
    #new entry for titles starts with Mrs
    data_set['Mrs'] = np.where(data_set['Titles'] == 2, 1,0)
    #new entry for other titles
    data_set['Others'] = np.where(data_set['Titles'] == 3, 1,0)
    #drop the remaining 'Embarked' column
    data_set.drop('Titles',axis = 1,inplace = True)


# In[42]:


IDs = test['PassengerId']
# normalize Parch, SibSp and PassengerId
for data_set in combined_data_set:
      data_Parch =(data_set['Parch']-data_set['Parch'].min())/(data_set['Parch'].max()-data_set['Parch'].min())
      data_set['Parch'] = data_Parch
      data_SibSp =(data_set['SibSp']-data_set['SibSp'].min())/(data_set['SibSp'].max()-data_set['SibSp'].min())
      data_set['SibSp'] = data_SibSp
      #data_PassengerId =(data_set['PassengerId']-data_set['PassengerId'].min())/(data_set['PassengerId'].max()-data_set['PassengerId'].min())
      #data_set['PassengerId'] = data_PassengerId


# In[43]:


# fill the missing Fare based on the Passanger Class
for data_set in combined_data_set:
    data_set['Fare'].fillna(data_set.groupby('Pclass')['Fare'].transform('median'), inplace = True)
for data_set in combined_data_set:
    data_set.loc[data_set['Fare'] <= 17, 'Fare'] = 0    
    data_set.loc[data_set['Fare'] > 17 & (data_set['Fare'] <= 30),'Fare'] = 1 
    data_set.loc[data_set['Fare'] > 30 & (data_set['Fare'] <= 100),'Fare'] = 2
    data_set.loc[data_set['Fare'] > 100 ,'Fare'] = 3
    #new entry for Cheapest Fare
    data_set['0_Fare'] = np.where(data_set['Fare'] == 0, 1,0)
    #new entry for Second Cheapest Fare
    data_set['1_Fare'] = np.where(data_set['Fare'] == 1, 1,0)
    #new entry for Third Cheapest Fare
    data_set['2_Fare'] = np.where(data_set['Fare'] == 2, 1,0)
    #new entry for Forth Cheapest Fare
    data_set['3_Fare'] = np.where(data_set['Fare'] == 3, 1,0)
    #drop the remaining 'Embarked' column
    data_set.drop('Fare',axis = 1,inplace = True)


# In[44]:


#new entries for Pclass
for data_set in combined_data_set:
    #new entry for passengers holds 1st class
    data_set['1_class'] = np.where(data_set['Pclass'] == 1, 1,0)
    #new entry for passengers hold  2nd class
    data_set['2_class'] = np.where(data_set['Pclass'] == 2, 1,0)
    #new entry for passengers holds 3rd class
    data_set['3_class'] = np.where(data_set['Pclass'] == 3, 1,0)
    #drop the remaining 'Embarked' column
    data_set.drop('Pclass',axis = 1,inplace = True)


# In[45]:


# convert Cabin to numerical data that we can use


# In[46]:


for data_set in combined_data_set:
    data_set.drop(['Cabin','Ticket','Name','PassengerId','Age','SibSp','Parch','1_Fare','2_Fare','Q'], axis = 1, inplace = True)
    #data_set.drop(data_set.columns.difference(['Fare','Age','Sex','Parch']), axis = 1, inplace = True)


# In[ ]:





# In[47]:


model = RandomForestClassifier(n_estimators = 100)
model.fit(X,Y)


# In[66]:


from sklearn.metrics import accuracy_score

accuracy_score(Y, model.predict(X))


# In[75]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
GBayes_clf = GaussianNB()
GBayes_clf.fit(X, Y)

print (GBayes_clf.score(X,Y))


# In[76]:


from sklearn.tree import DecisionTreeClassifier
    
tree_clf = DecisionTreeClassifier(max_depth = 5)
tree_clf.fit(X,Y)

print(tree_clf.score(X, Y))


# In[92]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)
clf.fit(X,Y)

print(clf.score(X,Y))


# In[ ]:


prediction = model.predict(test)
prediction = {'PassengerId': IDs, 'Survived': prediction}
pd.DataFrame(prediction).to_csv('prediction_3.csv', index = False)


# In[ ]:


X


# In[ ]:




