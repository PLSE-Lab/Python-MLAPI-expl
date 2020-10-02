#!/usr/bin/env python
# coding: utf-8

# # Predicting Titanic Survivors + EDA
# At 2:20am on 15th April 1912, the British ocean liner Titanic sank into the North Atlantic Ocean about 400 miles south of Newfoundland, Canada. The massive ship, which carried 2,200 passengers and crew, had struck an iceberg two and half hours before.
# 
# Everybody said it was 'unsinkable'; no-one predicted its tragic demise. For a moving depiction of the Titanic's sinking I recommend the 1959 movie ["A Night to Remember"](https://www.imdb.com/title/tt0051994/).
# 
# In this notebook I explore the famous Titanic dataset, accessed through the [Kaggle page](https://www.kaggle.com/c/titanic).
# I use a range of visualisations and test several Machine Learning models to predict survivors. 
# 
# ![alt text](https://www.gjenvick.com/Images600/Titanic/Photos/RMS-Titanic-1911-500.jpg)
# 
# The RMS Titanic
# 
# Source: gjenvick.com

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
pd.options.display.max_columns = 100

from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv("/kaggle/input/titanic/test.csv")
data = pd.concat([train.drop('Survived',axis=1),test])


# In[ ]:


train.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data.isnull(),cmap="viridis",yticklabels=False,cbar=False)


# In[ ]:


train.info()


# In[ ]:


test.info()


# # Step 1: Data Cleaning
# * Before I explore the data with some visualisations, I'll clean up the dataset, dealing with typos and dropping redundant variables.

# In[ ]:


train['PassengerId'].value_counts()


# In[ ]:


train['Name'].value_counts()


# In[ ]:


train['Cabin'].value_counts()


# In[ ]:


train['Ticket'].value_counts()


# The above 4 categories are redundant features for my purposes:
# * 'PassengerId', 'Cabin' and 'Ticket' are all arbitrary IDs that add no present insight;
# * However I will keep 'PassengerId' so that I can submit results of my model at the end of notebook
# * 'Name' does not tell me much that I can't derive from 'Sex' or 'Age'
# 
# I will therefore drop these 4 columns.

# In[ ]:


train = train.drop(['Cabin','Ticket','Name'],axis=1)
test = test.drop(['Cabin','Ticket','Name'],axis=1)


# In[ ]:


# rename columns to be more descriptive
train.rename(columns={"Pclass": "PClass", "Parch": "ParCh"},inplace=True)
test.rename(columns={"Pclass": "PClass", "Parch": "ParCh"},inplace=True)


# # Step 2: EDA
# * Before I clean the dataset and prepare it for modelling, I want to do some initial Exploratory Data Analysis to get an idea of which factors are most indicative of survival/fatality.

# In[ ]:


# define default figsize helper function

def set_figsize():
    '''
    Sets default figsize to 12x8
    '''
    plt.figure(figsize=[12,8])


# In[ ]:


# define default legend helper function

def legend_survived():
    '''
    Plots legend with Not survived & Survived
    '''
    plt.legend(['Did not survive','Survived'],loc='best')


# In[ ]:


# create subsets of survived vs not_survived for hue in plots

survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]


# ## Discrete Variables
# Let's start with some plots for discrete variables (categorical variables)

# In[ ]:


set_figsize()
plt.title('Survival count by Sex')
sns.countplot('Sex',data=train,hue='Survived')
legend_survived()


# It is evident that male passengers are far less likely to survive than females.

# In[ ]:


set_figsize()
plt.title('Survival count by Passenger class')
sns.countplot('PClass',data=train,hue='Survived')
legend_survived()


# As we can see, passengers in 1st class are most likely to survive. Passengers in 3rd class are by far least likely to survive.

# In[ ]:


set_figsize()
plt.title('Survival count by Port of Embarkation')
sns.countplot('Embarked',data=train,hue='Survived')
legend_survived()


# In[ ]:


set_figsize()
plt.title('Survival count by Number of Siblings/Spouse')
sns.countplot('SibSp',data=train,hue='Survived')
plt.xlabel('Number of Siblings/Spouse')
legend_survived()


# In[ ]:


set_figsize()
plt.title('Survival count by Number of Parents/Children')
sns.countplot('ParCh',data=train,hue='Survived')
plt.xlabel('Number of Parents/Children')
plt.legend(['Did not survive','Survived'],loc='upper right')


# ## Continuous Variables
# Now I will plot the remaining continuous variables ('Age' is not technically continuous, but can be treated as such for plots)

# In[ ]:


plt.figure(figsize=(20,8))

ax1 = sns.kdeplot(not_survived['Fare'],shade=True)
ax1.set_xlim((0,150))

ax2 = sns.kdeplot(survived['Fare'],shade=True)
ax2.set_xlim((0,150))

legend_survived()
plt.title('Survival density by Fare')
plt.xlabel('Fare')
plt.ylabel('Density')


# On the whole, we see that passengers who paid a higher fare are more likely to have survived.

# In[ ]:


plt.figure(figsize=(20,8))

ax1 = sns.kdeplot(not_survived['Age'],shade=True)

ax2 = sns.kdeplot(survived['Age'],shade=True)

legend_survived()
plt.title('Survival density by Age')
plt.xlabel('Age')
plt.ylabel('Density')


# ### Insights on Age
# Surprisingly, Age does not seem to contribute very much to survival.
# 
# The main conclusion is that passengers in their early 20s are least likely to survive.
# 
# **The plot for 'Survived' is a bimodal distribution. This tells me that extra effort was made on board the Titanic to save the youngest passengers (between 0-5 years)**

# # Step 3: Preprocessing
# Now that I've cleaned the data and performed EDA, I'll prepare the data for modelling by creating dummy variables and ensuring all data is numerical.

# In[ ]:


# Imputing Missing Age Values: choose median due to outliers, which affect mean
train['Age'].fillna(train['Age'].median(),inplace=True)

# Imputing Missing Embarked Values
train['Embarked'].fillna(train['Embarked'].value_counts().index[0], inplace=True)

#Creating a dictionary to convert Passenger Class from 1,2,3 to 1st,2nd,3rd.
d = {1:'1st',2:'2nd',3:'3rd'}

#Mapping the column based on the dictionary
train['PClass'] = train['PClass'].map(d)

# Getting Dummies of Categorical Variables
cat_vars = train[['PClass','Sex','Embarked']]
dummies = pd.get_dummies(cat_vars,drop_first=True)

# Drop original cat_vars
train = train.drop(['PClass','Sex','Embarked'],axis=1)
# Concatenate dummies and train
train = pd.concat([train,dummies],axis=1)

# Check the clean version of the train data.
train.head()


# In[ ]:


# split features and label
X = train.drop(['Survived'],1)
y = train['Survived']

# Use train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ## GradientBoostingClassifier
# First I'll try with a GBC

# In[ ]:


# choose GBC
# make predictions
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(learning_rate=0.1,max_depth=3)
GBC.fit(X_train, y_train)
pred_GBC = GBC.predict(X_test)


# In[ ]:


# evaluate performance
from sklearn.metrics import confusion_matrix, classification_report
print("GBC results:\n")
print(confusion_matrix(y_test, pred_GBC))
print(classification_report(y_test,pred_GBC))


# * This gives us an accuracy of 80%

# ## Random Forest
# Now I'll train a Random Forest model

# In[ ]:


# Try RFC
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)
pred_RFC = RFC.predict(X_test)


# In[ ]:


# evaluate performance
print("RFC results:\n")
print(confusion_matrix(y_test, pred_RFC))
print(classification_report(y_test,pred_RFC))


# * This gives a lower accuracy of 79%

# ## Logistic Regression
# Now let's try logistic regression.

# In[ ]:


# Try logistic regression
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_train, y_train)
pred_logit = logit.predict(X_test)


# In[ ]:


# evaluate performance
print("Logistic Regression results:\n")
print(confusion_matrix(y_test, pred_logit))
print(classification_report(y_test,pred_logit))


# In[ ]:


# SVC
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)


# In[ ]:


# evaluate performance
print("SVC results:\n")
print(confusion_matrix(y_test, pred_svc))
print(classification_report(y_test,pred_svc))


# In[ ]:


# Imputing Missing Age Values: choose median due to outliers, which affect mean
test['Age'].fillna(test['Age'].median(),inplace=True)

# Imputing Missing Embarked Values
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Impute Embarked
test['Embarked'].fillna(test['Embarked'].value_counts().index[0], inplace=True)


# In[ ]:


#Creating a dictionary to convert Passenger Class from 1,2,3 to 1st,2nd,3rd.
d = {1:'1st',2:'2nd',3:'3rd'}

#Mapping the column based on the dictionary
test['PClass'] = test['PClass'].map(d)

# Getting Dummies of Categorical Variables
cat_vars = test[['PClass','Sex','Embarked']]
dummies = pd.get_dummies(cat_vars,drop_first=True)

# Drop original cat_vars
test = test.drop(cat_vars,axis=1)
# Concatenate dummies and train
test = pd.concat([test,dummies],axis=1)

idx = test[['PassengerId']]


# In[ ]:


preds = model.predict(test)
results = idx.assign(Survived=preds)
results.to_csv('GBC_submission.csv',index=False)


# In[ ]:




