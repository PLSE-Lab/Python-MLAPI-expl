#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
combine = [train_df, test_df]


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


train_df.describe() # show numerical features


# In[ ]:


train_df.describe(include=['O']) #show categorical features


# In[ ]:


#replace Sex with numeric value 0 - male, 1 - female
mappingSex = {'male': 0, 'female': 1}
mappingEmbarked = {'Q': 0, 'S': 1, 'C': 2}
train_df = train_df.replace({'Sex': mappingSex, 'Embarked': mappingEmbarked})
test_df = test_df.replace({'Sex': mappingSex, 'Embarked': mappingEmbarked})


# In[ ]:


#Add a column to indicate Age NaN
train_df['MissingAge'] = 0
test_df['MissingAge'] = 0

train_df['Age'].fillna(0, inplace=True)
train_df.loc[train_df.Age == 0, 'MissingAge'] = 1

test_df['Age'].fillna(0, inplace=True)
test_df.loc[test_df.Age == 0, 'MissingAge'] = 1

train_df['Embarked'].fillna(0, inplace=True)
test_df['Embarked'].fillna(0, inplace=True)

train_df.info()


# In[ ]:


plt.hist(train_df["Age"], bins=20)


# In[ ]:


#Add a column to indicate Fare NaN
train_df['MissingFare'] = 0
test_df['MissingFare'] = 0

train_df['Fare'].fillna(0, inplace=True)
train_df.loc[train_df.Fare == 0, 'MissingFare'] = 1

test_df['Fare'].fillna(0, inplace=True)
test_df.loc[test_df.Fare == 0, 'MissingFare'] = 1

train_df.info()


# In[ ]:


corr = train_df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
plt.figure(figsize = (16,5))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, 
            square=False, linewidths=.5)


# In[ ]:


#Missing_df = train_df[train_df['MissingAge'] == 1]
#print(Missing_df['Embarked'].value_counts())
#print('_'*40)
#train_df['Embarked'].value_counts()


# In[ ]:


train_df.describe(include=['O'])


# In[ ]:


train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)
test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)


# In[ ]:


X_train = train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Survived'], axis = 1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


X_train.describe()


# In[ ]:


# Initialise the Scaler 
scaler = StandardScaler() 
# To scale data 
print(scaler.fit(X_train))
print(scaler.mean_)
#std_scale = preprocessing.StandardScaler().fit(X_train)
#X_train.replace(std_scale.transform(X_train))
#scaler.transform(X_train) 
#X_train


# In[ ]:


X_train['Age'] = X_train['Age'].astype(int)
X_train['Fare'] = X_train['Fare'].astype(int)
X_train['Embarked'] = X_train['Embarked'].astype(int)

X_test['Age'] = X_test['Age'].astype(int)
X_test['Fare'] = X_test['Fare'].astype(int)
X_test['Embarked'] = X_test['Embarked'].astype(int)

X_train


# ToDo feature normalization

# In[ ]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_scaled, Y_train)
Y_pred = random_forest.predict(X_scaled_test)
random_forest.score(X_scaled, Y_train)
acc_random_forest = round(random_forest.score(X_scaled, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

