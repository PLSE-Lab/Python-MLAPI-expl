#!/usr/bin/env python
# coding: utf-8

# Kindly,provide feedback and help me to grow.
# Upvote if you like my analysis.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling

import seaborn as sns
sns.set_style('whitegrid')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data= pd.read_csv('../input/titanic/train.csv')

test_data= pd.read_csv('../input/titanic/test.csv')
test_data['Survived']= np.nan
full_data= pd.concat([train_data,test_data])


# In[ ]:


full_data.profile_report()


# # 1. Feature Engineering

# ## 1.1 Dealing with missing values

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(full_data.isnull(),yticklabels=False,cbar=False)


# Null values in Survived column are of the test dataset. Age and Cabin columns contain many missing values.

# In[ ]:


# Let's calculate percentages of missing values!
full_data.isnull().mean().sort_values(ascending = False)


# Embarked and Fare have less than 1% missing values.So, we will simply fill them with mode and median.
# To fill missing Age values I will find most correlated factor with age.

# In[ ]:


from statistics import mode
full_data["Embarked"] = full_data["Embarked"].fillna(mode(full_data["Embarked"]))


# In[ ]:


sns.heatmap(full_data.corr(),cmap='viridis')


# So, we will fill Age and Fare column with help of Pclass feature.

# In[ ]:


full_data['Fare'] = full_data.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


full_data['Age'] = full_data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


full_data['Cabin'].isna().sum()/len(full_data)


# Almost 3/4th data is missing in Cabin feature.So, we will drop this column.

# In[ ]:


full_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


full_data.info()


# ## 1.2 Converting categorical columns.

# Now we will convert categorical columns into numerical using dummy variables.

# In[ ]:


embarked = pd.get_dummies(full_data[['Embarked','Sex']],drop_first=True)
full_data = pd.concat([full_data,embarked],axis=1)


# We will drop PassengerId and Ticket column as it doesn't seem important.Name too is not of much significance but salutation can be of importance.

# In[ ]:


Name1 = full_data['Name'].apply(lambda x : x.split(',')[1])


# In[ ]:


full_data['Title'] = Name1.apply(lambda x : x.split('.')[0])


# In[ ]:


full_data['Title'].value_counts(normalize=True)*100


# Except first four titles all form less than 1% of the data.So, we will combine them into one category and then form dummy variables.

# In[ ]:


full_data['Title'] = full_data['Title'].replace([ ' Don', ' Rev', ' Dr', ' Mme',' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',' the Countess', ' Jonkheer', ' Dona'], 'Other')


# In[ ]:


full_data['Title'].unique()


# In[ ]:


embarked = pd.get_dummies(full_data['Title'],drop_first=True)
full_data = pd.concat([full_data,embarked],axis=1)


# In[ ]:


full_data.drop(['PassengerId','Name','Sex','Ticket','Title','Embarked'],axis=1,inplace=True)


# In[ ]:


full_data.info()


# Now, let's retrieve our training and test data. And then convert each feature to integer.

# In[ ]:


test = full_data[full_data['Survived'].isna()].drop(['Survived'], axis = 1)
train = full_data[full_data['Survived'].notna()]


# In[ ]:


train = train.astype(np.int64)
test = test.astype(np.int64)


# In[ ]:


train


# # 2.Exploratory data analysis

# In[ ]:


sns.countplot(x='Survived',data=train_data,hue='Sex')


# Somehow, those who survived had more ratio of females and vice versa.  ;)

# In[ ]:


sns.countplot(x='Survived',data=train_data,hue='Pclass')


# In[ ]:


sns.distplot(train['Age'],kde=False,color='darkred',bins=30)


# Mostly people on board were aged between 20-40.

# In[ ]:


sns.countplot(x='SibSp',data=train)


# Mostly people on board were without their siblings or spouse.

# In[ ]:


sns.countplot(x='Parch',data=train)

Mostly people on board were travelling alone.
# In[ ]:


train['Fare'].hist(color='green',bins=40,figsize=(12,6))
plt.xlabel('Fare')


# Fare seems to be mostly below 100.

# # 3. Applying Logistic Regression.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                    train['Survived'], test_size = 0.2, 
                                                    random_state = 2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(max_iter = 10000)
logisticRegression.fit(X_train, y_train)


# In[ ]:


predictions = logisticRegression.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))


# In[ ]:


print(classification_report(y_test, predictions))


# Let's improve our accuracy by using N-fold cross-validation.

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


kf = KFold(n_splits = 5)
cross_val_score(logisticRegression, train.drop('Survived', axis = 1),train['Survived'], cv = kf).mean()


# It has improved to 81%.
# 

# # 4. Applying deep neural network

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


model = Sequential()
model.add(Dense(units=12,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=20,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=20,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[ ]:


model.fit(x=X_train.values, 
          y=y_train.values, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[ ]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[ ]:


dnn_predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,dnn_predictions))


# In[ ]:


print(confusion_matrix(y_test,dnn_predictions))


# It provides less accuracy than logistic regression

# # 5. Applying Random Forest.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_pred))


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:


param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7]    
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
randomForest_CV = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)


# In[ ]:


grid_pred = randomForest_CV.predict(X_test)


# In[ ]:


print(classification_report(y_test,grid_pred))


# In[ ]:


print(confusion_matrix(y_test,grid_pred))


# In[ ]:


randomForest_CV.best_params_


# # 6. Submitting predictions.

# In[ ]:


test['Survived'] = logisticRegression.predict(test)


# In[ ]:


test['PassengerId'] = test_data['PassengerId']


# In[ ]:


test[['PassengerId', 'Survived']].to_csv('lm_submission.csv', index = False)


# In[ ]:




