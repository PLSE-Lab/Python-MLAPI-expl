#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

import random

file_path = '../input/titanic/train.csv'
file_path_test = '../input/titanic/test.csv'

data = pd.read_csv(file_path)
X_test = pd.read_csv(file_path_test)
y = data.Survived
data = data.drop('Survived',1)
data = data.drop('Name',1)
data = data.drop('PassengerId',1)
data = data.drop('Cabin',1)
data = data.drop('Ticket',1)

data['SibSp'] = data['SibSp']+data['Parch']
data = data.drop('Parch',1)

data['Sex'][data['Sex']=='male'] = 0
data['Sex'][data['Sex']=='female'] = 1

data['Embarked'][data['Embarked']=='S'] = 0
data['Embarked'][data['Embarked']=='C'] = 1
data['Embarked'][data['Embarked']=='Q'] = 2

X_train,XX_test,Y_train,Y_test=train_test_split(data,y,test_size=0.3,random_state=random.seed())

#titanic_model = ExtraTreesRegressor(n_estimators=10, random_state=0)
titanic_model = XGBClassifier()
#titanic_model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)


# In[ ]:


#y = data.Survived
PassengerId = X_test['PassengerId']
#data = data.drop('Survived',1)
X_test = X_test.drop('Name',1)
X_test = X_test.drop('PassengerId',1)
X_test = X_test.drop('Cabin',1)
X_test = X_test.drop('Ticket',1)

X_test['SibSp'] = X_test['SibSp']+X_test['Parch']
X_test = X_test.drop('Parch',1)

X_test['Sex'][X_test['Sex']=='male'] = 0
X_test['Sex'][X_test['Sex']=='female'] = 1

X_test['Embarked'][X_test['Embarked']=='S'] = 0
X_test['Embarked'][X_test['Embarked']=='C'] = 1
X_test['Embarked'][X_test['Embarked']=='Q'] = 2


# In[ ]:


my_imputer = SimpleImputer(strategy='mean')

imputed_XX_train = my_imputer.fit_transform(X_train)
imputed_XX_valid = my_imputer.transform(XX_test)

imputed_X_train = my_imputer.fit_transform(data)
imputed_X_valid = my_imputer.transform(X_test)


# In[ ]:


titanic_model.fit(imputed_X_train, y)
Y_predi =  titanic_model.predict(imputed_X_valid)
print(titanic_model.score(imputed_XX_valid,Y_test))
submission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': Y_predi })
submission.to_csv("titanic_submition_8.0.csv", index=False)


# In[ ]:


imputed_X_valid.shape


# In[ ]:


from sklearn.metrics import mean_absolute_error

predictions = titanic_model.predict(imputed_XX_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, Y_test)))
#print(predictions)
#print(Y_test)


# from sklearn.metrics import mean_absolute_error
# 
# predictions = titanic_model.predict(imputed_X_valid)
# print("Mean Absolute Error: " + str(mean_absolute_error(predictions, Y_test)))
# print(predictions)
# print(Y_test)

# file_path = '../input/titanic/test.csv'
# data = pd.read_csv(file_path)
# 
# data = data.fillna(train_mode)
# 
# 
# # Make copy to avoid changing original data 
# data_sub = data.copy()
# 
# 
# # Apply label encoder to each column with categorical data
# label_encoder = LabelEncoder()
# for col in object_cols:
#     data_sub[col] = label_encoder.fit_transform(data[col])  
#     
# data_sub = pd.DataFrame(my_imputer.fit_transform(data_sub))
# 
# sub = titanic_model.predict(data_sub)
# submission = pd.DataFrame({ 'PassengerId': data.PassengerId.values, 'Survived': sub })
# submission.to_csv("titanic_submition.csv", index=False)
