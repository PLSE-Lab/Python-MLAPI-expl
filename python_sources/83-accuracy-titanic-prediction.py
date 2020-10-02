#!/usr/bin/env python
# coding: utf-8

# # Titanic EDA and Predictions

# In this kernel we will do some basic exploratory data analysis and will make some predictions about the chances of the survival of a person in Titanic.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.shape


# ### Basic Data Visualisation

# Let's find out the number of male and female passengers in the ship

# In[ ]:


p = sns.countplot(data=train_data, x='Sex')


# Embark feature tells us the Port of Embark. The following plot shows the number of passengers from different ports

# In[ ]:


p = sns.countplot(data=train_data, x='Embarked')
_ = plt.title('C = Cherbourg, Q = Queenstown, S = Southampton')


# The following plot shows the number of people who surrvived and who didn't survived the shipwrcek

# In[ ]:


p = sns.countplot(data=train_data, x='Survived')
_ = plt.title('0 = No, 1 = Yes')


# Passengers in different classes. Pclass feature is a proxy for socio-economic status

# In[ ]:


p = sns.countplot(data=train_data, x='Pclass')
_ = plt.title('1 = 1st, 2 = 2nd, 3 = 3rd')


# ### Using XGBoost Classifier

# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


model = XGBClassifier()


# In[ ]:


train_data["Sex"] = train_data["Sex"].fillna("NA")
train_data["Embarked"] = train_data["Embarked"].fillna("C")
test_data["Sex"] = test_data["Sex"].fillna("NA")
test_data["Embarked"] = test_data["Embarked"].fillna("C")
train_data[['Pclass', 'Age', 'SibSp', 'Fare']] = train_data[['Pclass', 'Age', 'SibSp', 'Fare']].fillna(0)
test_data[['Pclass', 'Age', 'SibSp', 'Fare']] = test_data[['Pclass', 'Age', 'SibSp', 'Fare']].fillna(0)


# In[ ]:


genders = {'male': 0, 'female': 1, 'NA': 2}
embarks = {'C': 0, 'Q': 1, 'S': 2,}
train_data['Sex'] = train_data['Sex'].apply(lambda x: genders[x])
train_data['Embarked'] = train_data['Embarked'].apply(lambda x: embarks[x])
test_data['Sex'] = test_data['Sex'].apply(lambda x: genders[x])
test_data['Embarked'] = test_data['Embarked'].apply(lambda x: embarks[x])


# In[ ]:


X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked', 'Fare']]
Y = train_data['Survived']


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)


# In[ ]:


trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:


model.fit(trainX, trainY)


# In[ ]:


predict = model.predict(testX)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(predict, testY)


# ### Using Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model_f = RandomForestClassifier()


# In[ ]:


model_f.fit(trainX, trainY)


# In[ ]:


predict_f = model_f.predict(testX)


# In[ ]:


accuracy_score(predict_f, testY)


# ### Using Support Vector Machines

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model_svm = SVC(gamma='scale', decision_function_shape='ovo')


# In[ ]:


model_svm.fit(trainX, trainY)


# In[ ]:


predict_svm = model_svm.predict(testX)


# In[ ]:


accuracy_score(predict_f, testY)


# ### Using Neural Networks

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


model_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# In[ ]:


model_nn.fit(trainX, trainY)


# In[ ]:


predict_nn = model_nn.predict(testX)


# In[ ]:


accuracy_score(predict_nn, testY)


# ### Making Predictions

# In[ ]:


test_df = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked', 'Fare']]
test_df = sc.transform(test_df)
final_predictions = model_svm.predict(test_df)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": final_predictions
    })
submission.to_csv('submission.csv', index=False)

