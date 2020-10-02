#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data['Type'] = 'train'
test_data['Type'] = 'test'


# In[ ]:


explr_data = train_data.append(test_data)


# In[ ]:


explr_data.head()


# In[ ]:


n_row = explr_data.shape[0]
for col in explr_data.columns:
    print(col, explr_data[col].isna().sum()/n_row*100)


# In[ ]:


explr_data = explr_data.drop(columns=['Cabin'])


# In[ ]:


explr_data['Sex'] = explr_data['Sex'].replace('male', 1).replace('female', 0)


# In[ ]:


explr_data['family_size'] = explr_data['Parch'] + explr_data['SibSp'] + 1


# In[ ]:


explr_data['family_size'].plot(kind='hist')


# In[ ]:


explr_data['family'] = 'Alone'
explr_data.loc[explr_data['family_size'] > 1, 'family'] = 'Small'
explr_data.loc[explr_data['family_size'] > 4, 'family'] = 'Big'


# In[ ]:


explr_data = explr_data.drop(columns=['Parch', 'SibSp', 'family_size'])


# In[ ]:


explr_data['family'].value_counts().plot(kind='bar')


# In[ ]:


explr_data = pd.concat([explr_data, pd.get_dummies(explr_data['family'], drop_first=True)], axis=1)
explr_data = explr_data.drop(['family'], axis=1) 


# In[ ]:


explr_data['Title'] = explr_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
explr_data = explr_data.drop('Name', axis=1)


# In[ ]:


mapping = {'Mlle': 'Miss', 
           'Ms': 'Miss', 
           'Mme': 'Mrs',
           'Major': 'Other', 
           'Col': 'Other', 
           'Dr' : 'Other', 
           'Rev' : 'Other',
           'Capt': 'Other', 
           'Jonkheer': 'Royal',
           'Sir': 'Royal', 
           'Lady': 'Royal', 
           'Don': 'Royal',
           'Countess': 'Royal', 
           'Dona': 'Royal'}
explr_data.replace({'Title': mapping}, inplace=True)
titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']


# In[ ]:


for title in titles:
    explr_data.loc[explr_data['Title'] == title, 'Age'].plot(kind='density', title='Age, Title='+title)
    plt.show()


# In[ ]:


median_age = explr_data[['Age', 'Title']].groupby(['Title']).median()


# In[ ]:


age_mask = explr_data['Age'].isna()
explr_data.loc[age_mask, 'Age'] = median_age.loc[explr_data.loc[age_mask, 'Title'], 'Age'].tolist()


# In[ ]:


explr_data['Age'].plot(kind='density')


# In[ ]:


explr_data['Fare'].plot(kind='hist')


# In[ ]:


avg_fare = explr_data.loc[explr_data['Pclass'] == 3, 'Fare'].mean()


# In[ ]:


explr_data.loc[explr_data['Fare'].isna(), 'Fare'] = avg_fare


# In[ ]:


n_row = explr_data.shape[0]
for col in explr_data.columns:
    print(col, explr_data[col].isna().sum()/n_row*100)


# In[ ]:


explr_data.loc[:, 'Child'] = 1
explr_data.loc[explr_data['Age']>=18, 'Child'] = 0


# In[ ]:


explr_data = explr_data.drop(['Age'], axis=1)


# In[ ]:


explr_data.head()


# In[ ]:


explr_data = explr_data.drop(['Title', 'Ticket'], axis=1)


# In[ ]:


embarked_dummy = pd.get_dummies(explr_data['Embarked'], drop_first=True)
explr_data = pd.concat([explr_data, embarked_dummy], axis=1)


# In[ ]:


train_data = explr_data.loc[explr_data['Type'] == 'train',]
test_data = explr_data.loc[explr_data['Type'] == 'test',]


# In[ ]:


train_data = train_data.drop(['Type', 'Embarked'],axis=1)
test_data = test_data.drop(['Type', 'Embarked'], axis=1)


# In[ ]:


train_col_list = ['Fare', 'Pclass', 'Sex', 'Big', 'Small', 'Child', 'Q', 'S']


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train_data[train_col_list], train_data['Survived'], test_size=0.2)


# ## RandomForestClassifier

# In[ ]:


model1 = RandomForestClassifier(random_state=2, min_samples_leaf=10)
model1.fit(X_train, y_train)
print(accuracy_score(y_val, model1.predict(X_val)))


# In[ ]:


export_graphviz(model1.estimators_[5], out_file='tree.dot',
               feature_names=X_train.columns,
               class_names=list(y_train.unique().astype(str)),
               rounded=True, proportion=False,
               precision=2, filled=True)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
Image(filename='tree.png')


# In[ ]:


model1.fit(X_val[train_col_list], y_val)
y_pred = pd.Series(data=model1.predict(test_data[train_col_list]).astype(int), name='Survived')
rfc_submission = pd.concat([test_data['PassengerId'], y_pred], axis=1)
rfc_submission.to_csv('rfc_submission.csv', index=False)


# In[ ]:


model2 = LogisticRegression()
model2.fit(X_train, y_train)
print(accuracy_score(y_val, model2.predict(X_val)))


# In[ ]:


model2.fit(X_val[train_col_list], y_val)
y_pred2 = pd.Series(data=model2.predict(test_data[train_col_list]).astype(int), name='Survived')
lgr_submission = pd.concat([test_data['PassengerId'], y_pred2], axis=1)
lgr_submission.to_csv('lgr_submission.csv', index=False)

