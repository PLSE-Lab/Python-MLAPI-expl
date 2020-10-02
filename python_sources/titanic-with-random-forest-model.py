#!/usr/bin/env python
# coding: utf-8

# # Notebook for the Titanic Competition
# 
# I will use the Random Forest Model in this challenge. If you have any suggestion, leave a comment!
# 
# 
# ## *Summary:*
# * Exploratory Data Analysis (EDA)
# * Data treatment (handling missing values, cleaning data)
# * Training the model
# * Submitting the solution

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
import warnings
from statistics import mode
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from copy import deepcopy


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# # Exploratory Data Analysis (EDA)

# In[ ]:


train.head()


# | Variable name | Description |
# | --- | --- |
# | PassengerId | Survived (1) or died (0) |
# | Pclass | Passenger's class |
# | Name | Passenger's name |
# | Sex | Passenger's sex |
# | Age | Passenger's age |
# | SibSp | Number of siblings/spouses aboard |
# | Parch | Number of parents/children aboard |
# | Ticket | Ticket number |
# | Fare | Fare |
# | Cabin | Cabin |
# | Embarked | Port of embarcation |

# Let's first visualize null values on our training set on graph.

# In[ ]:


plt.style.use('seaborn')
plt.figure(figsize=(10,5))
sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma')
plt.title('Null Values in Training Set');


# Cabin data has to much null values to be interesting. It would not make any sense to replace these null values since they are too numerous. Let's just ignore this column!

# The 'Name' column does not seem interesting as well since it will be difficult to analyse it in order to determine the influence of the passenger's name on his survival.
# 
# It is the same for the 'Ticket' column.
# 
# Let's just get rid of these columns to clear the data as early as possible!

# In[ ]:


train.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)
test.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# Let's visualize the importance of the different parameters for the survival of the passengers.

# ## Pclass

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train.Survived)
plt.title('Number of passengers Survived');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Pclass', data=train)
plt.title('Number of passengers Survived');


# In[ ]:


pclass1 = train[train.Pclass == 1]['Survived'].value_counts(normalize=True).values[0]*100
pclass2 = train[train.Pclass == 2]['Survived'].value_counts(normalize=True).values[1]*100
pclass3 = train[train.Pclass == 3]['Survived'].value_counts(normalize=True).values[1]*100


print("Pclass-1: {:.1f}% People Survived".format(pclass1))
print("Pclass-2: {:.1f}% People Survived".format(pclass2))
print("Pclass-3: {:.1f}% People Survived".format(pclass3))


# It seems that third class passengers were a lot more likely to die than the other passengers. On the contrary, first class passengers were more likely to survive.
# 
# *Conclusion: Pclass is a relevant information to decide whether the passenger will survive of not.* 

# ## Sex

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train.Survived)
plt.title('Number of passengers Survived');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Sex', data=train)
plt.title('Number of passengers Survived');


# It looks like women were a lot more likely to survive the accident than men.
# 
# *Conclusion: Sex is a relevant information to decide whether the passenger will survive of not.* 

# ## Age

# In[ ]:


train['Age'].hist(bins=40)
plt.title('Age Distribution');


# In[ ]:


# set plot size
plt.figure(figsize=(15, 3))

# plot a univariate distribution of Age observations 
sns.distplot(train[(train["Age"] > 0)].Age, kde_kws={"lw": 3}, bins = 50)

# set titles and labels
plt.title('Distrubution of passengers age',fontsize= 14)
plt.xlabel('Age')
plt.ylabel('Frequency')
# clean layout
plt.tight_layout()


# **Age column has a non-uniform data and many outliers.**
# 
# **Outlier:** An outlier is an observation that lies an abnormal distance from other values in a random sample from a population. 

# In[ ]:


plt.figure(figsize=(15,5))

#Draw a box plot to show Age distributions with respect to survival status
sns.boxplot(y='Survived', x='Age', data=train, palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')

#Add a scatterplot for each category
sns.stripplot(y='Survived', x='Age', data=train, palette=["#3f3e6fd1", "#85c6a9"], linewidth = 0.6, orient = 'h')

plt.yticks(np.arange(2), ['Drowned', 'Survived'])
plt.title('Age distribution grouped by surviving status')
plt.ylabel('Surviving status')
plt.tight_layout()


# It seems that younger passengers had a slightly better chance of surviving than the older ones.
# 
# *Conclusion: Age seems to have a slight influence on the survival of the passenger.* 

# ## Number of Siblings/Spouses aboard

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['SibSp'])
plt.title('Number of siblings/spouses aboard');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='SibSp', data=train)
plt.legend(loc='right')
plt.title('Number of siblings/spouses aboard');


# Passengers with no sibling or spouse on board were more likely to drown. 
# 
# *Conclusion: SibSp is a relevant information to decide whether the passenger will survive of not.* 

# ## Number of parents/children aboard

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['Parch'])
plt.title('Number of parents/children aboard');

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Parch', data=train)
plt.legend(loc='right')
plt.title('Number of parents/children aboard');


# Passengers with no parents or children on board were more likely to drown. 
# 
# *Conclusion: Parch is a relevant information to decide whether the passenger will survive of not.* 

# ## Fare

# In[ ]:


# set plot size
plt.figure(figsize=(15, 3))

# plot a univariate distribution of Age observations 
sns.distplot(train[(train["Fare"] > 0)].Fare, kde_kws={"lw": 3}, bins = 50)

# set titles and labels
plt.title('Distrubution of fare',fontsize= 14)
plt.xlabel('Fare')
plt.ylabel('Frequency')
# clean layout
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(15,5))

#Draw a box plot to show Age distributions with respect to survival status
sns.boxplot(y='Survived', x='Fare', data=train, palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')

#Add a scatterplot for each category
sns.stripplot(y='Survived', x='Fare', data=train, palette=["#3f3e6fd1", "#85c6a9"], linewidth = 0.6, orient = 'h')

plt.yticks(np.arange(2), ['Drowned', 'Survived'])
plt.title('Fare distribution grouped by surviving status')
plt.ylabel('Surviving status')
plt.tight_layout()


# The higher the fare, the more likely to survive was the passenger!
# 
# *Conclusion: Fare is a relevant information to decide whether the passenger will survive of not.* 

# ## Port of embarcation

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['Embarked'])
plt.title('Name of Port of embarkation')

plt.subplot(1,2,2)
sns.countplot(x='Survived', hue='Embarked', data=train)
plt.legend(loc='right')
plt.title('Name of passenger Survived');


# In[ ]:


embark1 = train[train.Embarked == 'S']['Survived'].value_counts(normalize=True).values[1]*100
embark2 = train[train.Embarked == 'C']['Survived'].value_counts(normalize=True).values[0]*100
embark3 = train[train.Embarked == 'Q']['Survived'].value_counts(normalize=True).values[1]*100


print("S: {:.1f}% People Survived".format(embark1))
print("C: {:.1f}% People Survived".format(embark2))
print("Q: {:.1f}% People Survived".format(embark3))


# Passengers who embarked in Cherbourg seemed more likely to survive!
# 
# *Conclusion: Embarked is a relevant information to decide whether the passenger will survive of not.* 

# ## Correlation Heatmap

# In[ ]:


sns.heatmap(train.corr(), annot=True);


# This heatmap allows us to easily spot the correlations between the different columns.
# 
# For example, there is a strong negative correlation between Pclass avec Fare. That was expected since first class passengers usually pay a higher fare than third class ones.

# # Data treatment

# Firstly, we need to replace the null values in the Age column.
# 
# As we saw in the correlation heatmap, Age was most correlated with Pclass (in absolute value). So we will replace missing age values with median age for the passenger's Pclass.

# In[ ]:


train.loc[train.Age.isnull(), 'Age'] = train.groupby('Pclass').Age.transform('median')
test.loc[test.Age.isnull(), 'Age'] = test.groupby('Pclass').Age.transform('median')


# In[ ]:


train.Embarked.value_counts()


# As maximum values in train set is 'S', let's replace null values by 'S'.

# In[ ]:


train['Embarked'] = train['Embarked'].fillna(mode(train['Embarked']))
test['Embarked'] = test['Embarked'].fillna(mode(test['Embarked']))


# Fare was most correlated with Pclass, so we will replace missing values of Fare by the median fare for the passenger's Pclass.

# In[ ]:


train.loc[train.Fare.isnull(), 'Fare'] = train.groupby('Pclass').Fare.transform('median')
test.loc[test.Fare.isnull(), 'Fare'] = test.groupby('Pclass').Fare.transform('median')


# Now, we just need to replace Sex and Embarked values by numbers in order to use it correctly.
# 
# Concerning the sex, we can replace 'male' by 0 and 'female' by 1.

# In[ ]:


train['Sex'][train['Sex']=='male'] = 0
train['Sex'][train['Sex']=='female'] = 1

test['Sex'][test['Sex']=='male'] = 0
test['Sex'][test['Sex']=='female'] = 1


# Concerning the port of embarcation, we will encode the data with OneHotEncoder technique.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(train[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
train = train.join(temp)
train.drop(columns='Embarked', inplace=True)

temp = pd.DataFrame(encoder.transform(test[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
test = test.join(temp)
test.drop(columns='Embarked', inplace=True)


# In[ ]:


train.head()


# **Dataset is completely ready now!**

# # Training the model

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'], test_size = 0.2, random_state=2)


# ## Scale the data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# We must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)


# I will use the Random Forest algorithm for this classification problem.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(random_state=2)


# ## Hyperparameter Tuning
# 
# Below, we set the hyperparameter grid with 4 lists of values:
# 
# 'criterion': A function which measures the quality of a split.
# 'n_estimators': The number of trees of our random forest.
# 'max_features': The number of features to choose when looking for the best way of splitting.
# 'max_depth': the maximul depth of a decision tree.

# In[ ]:


# Set our parameter grid
param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [300, 400],
    'max_features': ['auto', 'log2'],
    'max_depth' : [6, 7]    
}


# In[ ]:


from sklearn.model_selection import GridSearchCV

randomForest_CV = GridSearchCV(estimator = rfclf, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)

randomForest_CV.best_params_


# In[ ]:


rf_clf = RandomForestClassifier(random_state = 2, criterion = 'entropy', max_depth = 7, max_features = 'auto', n_estimators = 400)

rf_clf.fit(X_train, y_train)


# In[ ]:


predictions = rf_clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions) * 100


# # Submitting the solution

# In[ ]:


scaler = MinMaxScaler()

train_conv = scaler.fit_transform(train.drop(['Survived', 'PassengerId'], axis=1))
test_conv = scaler.transform(test.drop(['PassengerId'], axis = 1))


# In[ ]:


rf_clf = RandomForestClassifier(random_state = 2, criterion = 'entropy', max_depth = 7, max_features = 'auto', n_estimators = 400)

rf_clf.fit(train_conv, train['Survived'])


# In[ ]:


test2 = deepcopy(test)

test2['Survived'] = rf_clf.predict(test_conv)


# In[ ]:


test2[['PassengerId', 'Survived']].to_csv('MySubmissionRandomForest.csv', index = False)

