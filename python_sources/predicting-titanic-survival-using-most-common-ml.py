#!/usr/bin/env python
# coding: utf-8

# # Prediction of passengers survival for Titanic data
# 
# <img src="https://cbsnews1.cbsistatic.com/hub/i/2018/10/23/80b06a72-0d2f-4962-a3c1-ab0b1a18c9bc/screen-shot-2018-10-23-at-10-45-06-am.png" width=600>
# 
# ## Introduction
# This prediction is based on the most common supervised machine learning models. It includes some EDA, feature engineering, and modeling. 
# 
# Below we will try to confirm the following assumptions:
# 1. Were women more likely to survive?
# 1. Were people traveling with children more likely to survive?
# 1. Were young children more likely to survive?
# 1. Were people with higher class tickets more likely to survive?
# 
# *April, 2020*

# In[ ]:


#Getting all the packages we need: 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #statist graph package
import matplotlib.pyplot as plt #plot package

#plt.style.use('ggplot') #choosing favorite R ggplot stype
plt.style.use('bmh') #setting up 'bmh' as "Bayesian Methods for Hackers" style sheet


#loading ML packages:
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#input files directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## <a name="read"></a>Reading the dataset

# In[ ]:


# Read train data
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


print("train shape:",train.shape)
print("test shape :",test.shape)


# In[ ]:


test.head(5)


# In[ ]:


train.sample(5)


# >Our target column for prediction is "Survived"

# In[ ]:


#The mean of the target column:
round(np.mean(train['Survived']), 2)


# In[ ]:


train['Survived'].value_counts()


# In[ ]:


#Survival rate:

color = ('#F5736B', '#C7F35B')
plt.pie(train["Survived"].value_counts(), data = train, explode=[0.08,0], labels=("Not Survived", "Survived"), 
        autopct="%1.1f%%", colors=color, shadow=True, startangle=400, radius=1.6, textprops = {"fontsize":20})
plt.show();


# ## <a name="read"></a>Getting a feel of a dataset

# >Below we run a quick EDA on the most essential data points: Age, Gender, Social class, and family size.

# In[ ]:


train.describe()


# >We note from the data above:
# 
# * There are 891 entries in the train dataset. Caveat, not all columns in the dataset are complete.
# * The yongest passenger age is 4 months, the oldest - 80 years. Average Titanic passenger age is 29 years, while the median is 28 years. That said, we have age data for approximately 80% passengers.
# * The largest family had 6 people (children and parents) and 8 siblings. 
# * Approximatelly half of the passengers were with siblings and/or a spouse. And more than a half pasengers were travelling alone without children and/or parents.  

# ## <a name="read"></a>So, what features we can use for prediction?

# In[ ]:


h_labels = [x.replace('_', ' ').title() for x in 
            list(train.select_dtypes(include=['number', 'bool']).columns.values)]

fig, ax = plt.subplots(figsize=(10,6))
_ = sns.heatmap(train.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)


# >We don't have high correlations which might affect our prediction model. Besides Age, we also might need to look into Parents/Children and Fare values. 

# ### <a name="read"></a>Social class

# In[ ]:


#How many people from which social class were on Titanic?

sns.countplot(train['Pclass'])


# In[ ]:


#Let's group social class and survived data:

sns.countplot(x = 'Survived', hue = 'Pclass', data = train)

#0 - didn't survived
#1 - survived


# >We can see above that the highest survived score has the first class. The lowest - the third class.

# In[ ]:


plt.figure(figsize = (16, 8))

sns.distplot(train["Fare"])
plt.title("Fare Histogram")
plt.xlabel("Fare")
plt.show()


# ### <a name="read"></a>Gender

# In[ ]:


#Let's see the same survival chart for gender data: 

sns.countplot(x = 'Survived', hue = 'Sex', data = train)

#0 - didn't survived
#1 - survived


# >As we assumed, women have significantly higher survival rate.

# ### <a name="read"></a>Age

# In[ ]:


#Age distribution
plt.figure(figsize = (16, 8))

sns.distplot(train["Age"])
plt.title("Age Histogram")
plt.xlabel("Age")
plt.show()


# In[ ]:


#Let's group age and survived data:
plt.figure(figsize = (35, 8))

sns.countplot(x = 'Age', hue = 'Survived', data = train)

plt.title("Age Histogram")
plt.xlabel("Age")
plt.show()

#0 - didn't survived
#1 - survived


# In[ ]:


g = sns.FacetGrid(train, col = "Survived")
g.map(sns.distplot, "Age")
plt.show()


# ### <a name="read"></a>Family size

# In[ ]:


sns.countplot(train['SibSp'])


# In[ ]:


#Let's group family and survived data:
plt.figure(figsize = (15, 8))

sns.countplot(x = 'SibSp', hue = 'Survived', data = train)

plt.title("Siblings/Spouse Histogram")
plt.xlabel("Siblings/Spouse")
plt.show()

#0 - didn't survived
#1 - survived


# >Most of the Titanic passengers were travelling alone. The survival rate for people travelling with a spouse or siblings may seem to be slightly higher. 

# In[ ]:


#Let's group children/parents and survived data:
plt.figure(figsize = (12, 6))

sns.countplot(x = 'Parch', hue = 'Survived', data = train)

plt.title("Parents/Children Histogram")
plt.xlabel("Parents/Children")
plt.show()

#0 - didn't survived
#1 - survived


# >It seems that passengers travelling with children and/or parents have higher chances to survive.

# ## <a name="read"></a>Feature engineering or preparing our data for modeling

# In[ ]:


#Let's review the data types we have to work with:

test.info()


# * Numerical Features: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)
# * Categorical Features: Survived, Sex, Embarked, Pclass
# * Alphanumeric Features: Ticket, Cabin

# In[ ]:


# Let's look for missing values
train.isnull().sum().sort_values(ascending = False)


# In[ ]:


# And missing values for Test set:
test.isnull().sum().sort_values(ascending = False)


# > ### <a name="read"></a>Challenges we have to solve before data modeling:
# 1. Incomplete age data (20% missing values). Assuming age is an important attribute for survival, we will have to fill in the gaps.
# 1. Gender is categorical data, which we need to transfer to numerical for LR model.
# 1. Cabin data is very sparse. We won't be able to use it. 
# 1. Embarced data has some missing values in the Train dataset. We might have to fill them in before using it.
# 1. Fare data also has missing values. 
# 

# ### <a name="read"></a>Age values 

# In[ ]:


#We have to fill in missing age values in our dataset. We can use Median Titanic passenger age data for this, which is 28 (as we confirmed it above in EDA)

train['Age']=train['Age'].fillna('28')
test['Age']=train['Age'].fillna('28')


# ### <a name="read"></a>Embarked values

# In[ ]:


#Using the same logic again and filling in "S" for missing Embarked values:

train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = train['Embarked'].fillna('S')


# In[ ]:


# Convert 'Embarked' variable to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


# ### <a name="read"></a>Fare

# In[ ]:


#Our test set has one empty value, which we will fill in with the median:  
test['Fare']=train['Fare'].fillna('14')


# ### <a name="read"></a>Gender values

# In[ ]:


#Convert categorical Gender column to numerical data:

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1


# In[ ]:


#Let's validate we don't have empty values left: 

train.isnull().sum().sort_values(ascending = False)


# In[ ]:


#And, for the test set:

test.isnull().sum().sort_values(ascending = False)


# In[ ]:


#Dropping columns which we won't use:

train.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
test.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# In[ ]:


#Final check for the Train data:

train.head(5)


# In[ ]:


#And, for the test set:

test.head(5)


# >Looks clean. We can build prediction model now!

# ## <a name="read"></a>Modeling

# In[ ]:


#Running model for only 20% our test sample using test split feature:

x_train, x_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                    train['Survived'], test_size = 0.2, 
                                                    random_state = 2)


# In[ ]:


#Logistic Regression model:

logisticRegression = LogisticRegression(max_iter = 10000)
logisticRegression.fit(x_train, y_train)

# Predicting the values for Survived:
predictions = logisticRegression.predict(x_test)

#print(predictions)

acc_logreg = round(accuracy_score(predictions, y_test) * 100, 2)
print(acc_logreg)


# In[ ]:


#Decision Tree Classifier:

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)

# Predicting the values for Survived:
predictions = decisiontree.predict(x_test)

#print(predictions)

acc_decisiontree = round(accuracy_score(predictions, y_test) * 100, 5)
print(acc_decisiontree)


# In[ ]:


#Gaussian NB:

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)

# Predicting the values for Survived:
predictions = gaussian.predict(x_test)

#print(predictions)

acc_gaussian = round(accuracy_score(predictions, y_test) * 100, 5)
print(acc_gaussian)


# In[ ]:


#Support Vector Machines

svc = SVC(max_iter = 10000)
svc.fit(x_train, y_train)

# Predicting the values for Survived:
predictions = svc.predict(x_test)

#print(predictions)

acc_svc = round(accuracy_score(predictions, y_test) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC

linear_svc = LinearSVC(max_iter = 10000)
linear_svc.fit(x_train, y_train)

# Predicting the values for Survived:
predictions = linear_svc.predict(x_test)

#print(predictions)

acc_linear_svc = round(accuracy_score(predictions, y_test) * 100, 2)
print(acc_linear_svc)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 'Naive Bayes', 'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_logreg, acc_gaussian, acc_linear_svc, acc_decisiontree]})
models.sort_values(by='Score', ascending=False)


# ## <a name="read"></a>Prediction and submission - let's use the model with the highest score.

# In[ ]:


#set ids as PassengerId and predict survival 

ids = test['PassengerId']
print(len(ids))
predictions = logisticRegression.predict(test)


# In[ ]:


#set the output file:
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.tail(5)


# In[ ]:


output.to_csv('kaggle_titanic_submission.csv', index=False)
print("Successfull submission")

