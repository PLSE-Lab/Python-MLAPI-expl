#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head(20)


# In[ ]:


Family_tot = train_data["SibSp"] + train_data["Parch"]
train_data["Family_tot"] = Family_tot
train_data.head()


# In[ ]:


Family_test = test_data["SibSp"] + test_data["Parch"]
test_data["Family_tot"] = Family_test


# In[ ]:


sns.heatmap(train_data.corr(), cmap='Blues')


# In[ ]:


sns.heatmap(test_data.corr(), cmap='Blues')


# In[ ]:


num_correlation = train_data.select_dtypes(exclude='object').corr()
corr = num_correlation.corr()
print(corr['Survived'].sort_values(ascending=False))


# In[ ]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


Faremean=test_data.loc[:,"Fare"].mean()
print(Faremean)
test_data['Fare'].fillna(Faremean,inplace = True)


# In[ ]:


test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:





# In[ ]:


test_data.info()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass","Sex","Age", "Family_tot", "Fare"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

