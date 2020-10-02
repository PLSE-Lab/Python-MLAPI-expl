#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival
# Titanic is an accident happen at 1912. A lot of people lives has been lost in this disaster. 
# 
# We will explore on data that have been given by Kaggle below.
# 
# From this Kernel, we are going to an analyst this disaster data into machine learning to predict who survived or not. 
# 
# <img src="https://sites.duke.edu/perspective/files/2018/07/titanic-7.jpg">

# # Libraries

# In[ ]:


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# # Read Data

# In[ ]:


raw_data = pd.read_csv("../input/train.csv")
raw_test = pd.read_csv('../input/test.csv')


# # Data Exploration

# **Estimator**
# 
# This estimator will help to generator bar plot with percentages base on it category

# In[ ]:


def survival_estimator(x):
    return len(x[x==1])/len(x)*100.0


# **Data Columns**

# In[ ]:


print(raw_data.columns)


# **What is data type of each column**

# In[ ]:


raw_data.info()


# **See some start data**

# In[ ]:


raw_data.head(10)


# **Check Null Data**

# In[ ]:


print(raw_data.isnull().sum())
print("-"*10)
print(raw_data.isnull().sum()/raw_data.shape[0])


# We can see there are a lot of people that don't have Cabin record. So we won't use this colum

# **Corralation Of Data**

# In[ ]:


cor_matrix = raw_data.drop(columns=['PassengerId']).corr().round(2)
# Plotting heatmap 
fig = plt.figure(figsize=(12,12));
sns.heatmap(cor_matrix, annot=True, cmap='autumn');


# **Survival Count**

# In[ ]:


sns.countplot(x='Survived', data=raw_data)


# **Survival By Sex**

# In[ ]:


sns.barplot(x="Sex",y="Survived", data=raw_data, estimator=survival_estimator)


# We can see female have more chance to survive than man. As we know they evaluate female first.

# **Survival By Class**

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=raw_data, estimator=survival_estimator)


# The people who say at upper class have more chance to survived

# **Plot Ages**

# In[ ]:


plt.figure(figsize=(18, 30))
sns.countplot(y='Age', data=raw_data)


# **Survival by Age**

# In[ ]:


raw_data['Age'] = raw_data['Age'].dropna().astype(int)
sns.FacetGrid(raw_data, hue='Survived', aspect=4).map(sns.kdeplot, 'Age', shade= True).set(xlim=(0 , raw_data['Age'].max())).add_legend()


# **Age by group every 10 year old**

# In[ ]:


raw_data['AgeGroup'] = pd.cut(raw_data.Age, bins=np.arange(start=0, stop=90, step=10), include_lowest=True)
plt.figure(figsize=(18, 5))
sns.barplot(x='AgeGroup', y='Survived', data=raw_data, estimator=survival_estimator)


# Graph of survive and unsurvive is almost similar. So age is not the reason people survive. But even so we can see that children under 5 year old have chance to survived. Even so we can see children under 5 year old has more chance to be survived.

# In[ ]:


raw_data['IsChildren'] = np.where(raw_data['Age']<10, 1, 0)
sns.barplot(x='IsChildren', y='Survived', data=raw_data, estimator=survival_estimator)


# Now we see the chidren is mostly survived

# Handle NAN age

# In[ ]:


raw_data['IsAgeNull'] = np.where(np.isnan(raw_data['Age']), 1, 0)
raw_data['Age'].fillna((raw_data['Age'].mean()), inplace=True)
raw_data['Age'] = raw_data['Age'].round().astype(int)


# In[ ]:


raw_data['AgeLabel'] = pd.cut(raw_data['Age'], bins=np.arange(start=0, stop=90, step=10), labels=np.arange(start=0, stop=8, step=1), include_lowest=True)


# **Survival by number of sibling or spouse**

# In[ ]:


plt.figure(figsize=(18, 8))
sns.barplot(x='SibSp', y='Survived', data=raw_data, estimator=survival_estimator)


# **Survival by number of parent or children**

# In[ ]:


plt.figure(figsize=(18, 8))
sns.barplot(x='Parch', y='Survived', data=raw_data, estimator=survival_estimator)


# **Travel With Parent**

# **Family Size**

# In[ ]:


plt.figure(figsize=(18, 8))
raw_data['FamilySize'] = raw_data.apply (lambda row: row['SibSp']+row['Parch'], axis=1)
sns.barplot(x='FamilySize', y='Survived', data=raw_data, estimator=survival_estimator)


# From this we can see who travel with 1, 2 or 3 people have more chance to survive

# In[ ]:


raw_data['NoFamily'] = np.where(raw_data['FamilySize']==0, 1, 0)
raw_data['SmallFamily'] = np.where((raw_data['FamilySize']>0)&(raw_data['FamilySize']<4), 1, 0)
raw_data['MediumFamily'] = np.where((raw_data['FamilySize']>3)&(raw_data['FamilySize']<7), 1, 0)
raw_data['LargeFamily'] = np.where(raw_data['FamilySize']>=7, 1, 0)

fig, axes = plt.subplots(1, 4, figsize=(18, 8))
sns.barplot(x='NoFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[0])
sns.barplot(x='SmallFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[1])
sns.barplot(x='MediumFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[2])
sns.barplot(x='LargeFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[3])


# **Survival by fare**

# In[ ]:


plt.subplots(1,1,figsize=(18, 8))
sns.distplot(raw_data['Fare'])


# We need to group fare by each 50 dollar

# In[ ]:


raw_data['FareRange'] = pd.cut(raw_data.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, include_lowest=True)
plt.figure(figsize=(18, 8))
sns.countplot('FareRange', data=raw_data)


# In[ ]:


plt.figure(figsize=(18, 8))
sns.barplot(x='FareRange', y='Survived', data=raw_data, estimator=survival_estimator)


# Label the Fare

# In[ ]:


raw_data['FareLabel'] = pd.cut(raw_data.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, labels=np.arange(start=0, stop=11, step=1), include_lowest=True)


# We can see that people likely to survive if Fare is higher

# In[ ]:


raw_data['LowFare'] = np.where(raw_data['Fare']<=50, 1, 0)
sns.barplot(x='LowFare', y='Survived', data=raw_data, estimator=survival_estimator)


# Low Fare has less chance to survived

# In[ ]:


raw_data['HighFare'] = np.where(raw_data['Fare']>300, 1, 0)
sns.barplot(x='HighFare', y='Survived', data=raw_data, estimator=survival_estimator)


# High Fare mostly have chance to survived

# In[ ]:


def medium_fare(row):
    if(row['LowFare']==0 & row['HighFare']==0):
        return 1
    else:
        return 0
raw_data['MediumFare'] = raw_data.apply(medium_fare, axis=1)
sns.barplot(x='MediumFare', y='Survived', data=raw_data, estimator=survival_estimator)


# Medium Fare have bigger chance to survived

# **Survival by Embarked**

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=raw_data, estimator=survival_estimator)


# People Cherbourg from get a lot of chance to survive

# **Embarked And Sex**

# In[ ]:


plt.figure(figsize=(18, 8))
sns.FacetGrid(raw_data,size=5, col="Sex", row="Embarked", hue = "Survived").map(plt.hist, "Age", edgecolor = 'white').add_legend();


# * We can see most of the passengers boarding at Southampton. And all of them mostly male who less likely to survived
# * Most of passengers from Cherbourg is female that why they have a lot of percentage to survived
# * Few female boarding at Queenstown and most of them survived
# 
# So we can Embarked is not related to the survived.

# **Title**

# In[ ]:


raw_data['Title'] = raw_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
plt.figure(figsize=(18, 8))
print(raw_data['Title'].unique())

raw_data['Title'] = raw_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
raw_data['Title'] = raw_data['Title'].replace('Mlle', 'Miss')
raw_data['Title'] = raw_data['Title'].replace('Ms', 'Miss')
raw_data['Title'] = raw_data['Title'].replace('Mme', 'Mrs')

sns.barplot(x="Title", y="Survived", data=raw_data, estimator=survival_estimator)


# **Cabin**

# In[ ]:


print(raw_data['Cabin'].unique())


# In[ ]:


raw_data['NoCabin'] = np.where(raw_data['Cabin'].isnull(), 1, 0)
sns.barplot(x="NoCabin", y="Survived", data=raw_data, estimator=survival_estimator)


# # Data Preparation

# **One hot encoding for sex**

# In[ ]:


raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['Sex'])], axis=1)


# **One hot encoding for title**

# In[ ]:


raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['Title'], prefix='title')], axis=1)


# **One hot encoding for class**

# In[ ]:


raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['Pclass'], prefix='Pclass')], axis=1)


# In[ ]:


raw_data.head(10)


# In[ ]:


need_columns = [
    'female', 'male', 
    'Pclass_1', 'Pclass_2', 'Pclass_3', 
    'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_Rare', 
    'AgeLabel', 'IsAgeNull', 'IsChildren', 
    'FareLabel', 'SibSp', 'Parch', 'FamilySize', 
    'NoFamily', 'SmallFamily', 'MediumFamily', 'LargeFamily', 
    'LowFare', 'HighFare', 'MediumFare', 
    'NoCabin'
]
data = raw_data[need_columns]

x = data
y = raw_data.Survived


# # Find Best Params

# Grid Params

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
max_leaf_nodes = [2, 5, 8, 10, None]
criterion=['gini', 'entropy']

random_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap,
    'max_leaf_nodes': max_leaf_nodes
}
estimator = RandomForestClassifier()


# Tunning Params

# In[ ]:


rf_random = RandomizedSearchCV(
    estimator=estimator, 
    param_distributions=random_grid, 
    random_state=42, 
    n_jobs=-1
)
rf_random.fit(x, y)


# What is the Best params

# In[ ]:


best_params = rf_random.best_params_
print(best_params)


# The best score

# In[ ]:


best_score = rf_random.best_score_
print("best score {}".format(best_score))


# # Evaluate Model

# We use cross-validation to evaluate your model. You can check more from [here](https://www.kaggle.com/dansbecker/cross-validation).

# In[ ]:


test_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    criterion=best_params['criterion'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    bootstrap=best_params['bootstrap']
)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(test_model, x, y, cv=10, scoring='accuracy')


# In[ ]:


print(scores)
print("Mean Accuracy: {}".format(scores.mean()))


# # Training with the whole dataset

# In[ ]:


model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    criterion=best_params['criterion'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    bootstrap=best_params['bootstrap']
)
model.fit(x, y)


# # Submit Result

# In[ ]:


print(raw_test.isnull().sum())
print("-"*10)
print(raw_test.isnull().sum()/raw_test.shape[0])


# In[ ]:


raw_test['Fare'].fillna(raw_test['Fare'].mean(), inplace=True)
raw_test['NoCabin'] = np.where(raw_test['Cabin'].isnull(), 1, 0)

raw_test['IsChildren'] = np.where(raw_test['Age']<=10, 1, 0)
raw_test['IsAgeNull'] = np.where(np.isnan(raw_test['Age']), 1, 0)
raw_test['Age'].fillna(raw_test['Age'].mean(), inplace=True)
raw_test['Age'] = raw_test['Age'].round().astype(int)
raw_test['AgeLabel'] = pd.cut(raw_test['Age'], bins=np.arange(start=0, stop=90, step=10), labels=np.arange(start=0, stop=8, step=1), include_lowest=True)

raw_test['FamilySize'] = raw_test.apply (lambda row: row['SibSp']+row['Parch'], axis=1)
raw_test['LowFare'] = np.where(raw_test['Fare']<=50, 1, 0)
raw_test['HighFare'] = np.where(raw_test['Fare']>300, 1, 0)
raw_test['MediumFare'] = raw_test.apply(medium_fare, axis=1)
raw_test['FareLabel'] = pd.cut(raw_test.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, labels=np.arange(start=0, stop=11, step=1), include_lowest=True)

raw_test['NoFamily'] = np.where(raw_test['FamilySize']==0, 1, 0)
raw_test['SmallFamily'] = np.where((raw_test['FamilySize']>0)&(raw_test['FamilySize']<4), 1, 0)
raw_test['MediumFamily'] = np.where((raw_test['FamilySize']>3)&(raw_test['FamilySize']<7), 1, 0)
raw_test['LargeFamily'] = np.where(raw_test['FamilySize']>=7, 1, 0)

raw_test['Title'] = raw_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
raw_test['Title'] = raw_test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
raw_test['Title'] = raw_test['Title'].replace('Mlle', 'Miss')
raw_test['Title'] = raw_test['Title'].replace('Ms', 'Miss')
raw_test['Title'] = raw_test['Title'].replace('Mme', 'Mrs')

raw_test = pd.concat([raw_test, pd.get_dummies(raw_test['Sex'])], axis=1)
raw_test = pd.concat([raw_test, pd.get_dummies(raw_test['Title'], prefix='title')], axis=1)
raw_test = pd.concat([raw_test, pd.get_dummies(raw_test['Pclass'], prefix='Pclass')], axis=1)
data_test = raw_test[need_columns]


# In[ ]:


ids = raw_test['PassengerId']
predictions = model.predict(data_test)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index = False)
output.head(10)

