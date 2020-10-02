#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Let's start with loading several helpful packages 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# # 1. A simple logistic regression

# ## a. Use only numercal variables

# We begin with a simple logistic regression model.
# First, we need to split our data in train and test datasets. We will then create and train a logistic regression model on the training dataset and calculate its accuracy on the test dataset. 
# 
# Note that we use only the fields 'Pclass', 'Age',  'SibSp', 'Parch', 'Fare' because these fields are already numeric and logistic regression can only work with numbers. Fields that have non-numeric values (eg. 'Sex' with the values 'male', 'female') need special processing before can be used as we will see later.

# In[5]:


train_df = pd.read_csv('../input/train.csv')

# keep only the numerical columns
data = train_df['Survived Pclass Age SibSp Parch Fare'.split()].fillna(0)
X, y = data['Pclass Age SibSp Parch Fare'.split()], data['Survived']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)

lr_model = LogisticRegression()
lr_model.fit(Xtrain, ytrain)
predictions = lr_model.predict(Xtest)

print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))
print('Confusion matrix:')
print(confusion_matrix(predictions, ytest, labels=[0, 1]))


# Now it's time to use our model to make predictions on the final dataset provided by kaggle competition. 
# 
# **Take the file submission1a.csv created by the model and submit it to kaggle to get scored.**

# In[6]:


test_df = pd.read_csv('../input/test.csv')

# keep only the numerical columns and the PassengerId (we need it to create the submission file)
final = test_df['PassengerId Age Pclass SibSp Parch Fare'.split()].fillna(0)
pid, Xfinal = final['PassengerId'], final[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

lr_model.fit(X, y)

final_predictions = lr_model.predict(Xfinal)

if not os.path.isdir('./output'):
    os.mkdir('output')
results = pd.DataFrame({'PassengerId': pid, 'Survived': final_predictions})
print(results.shape)
results.to_csv('./output/submission1a.csv', index=False)


# ## b. Encode categorical variables

# Now we will use the categorical variables. But first, we need to convert them to numerical values, starting with sex which is the simplest, using the numerical value 0 for male and 1 for female

# In[7]:


train_df = pd.read_csv('../input/train.csv')


data = train_df['Survived Age Pclass Sex SibSp Parch Fare'.split()].fillna(0)
# encode sex as binary variable
data.Sex = data.Sex.apply(lambda x: 0 if x == 'male' else 1)

X, y = data['Age Pclass Sex SibSp Parch Fare'.split()], data['Survived']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)

lr_model = LogisticRegression()
lr_model.fit(Xtrain, ytrain)
predictions = lr_model.predict(Xtest)

print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))
print('Confusion matrix:')
print(confusion_matrix(predictions, ytest, labels=[0, 1]))


# Observe the improvement of the accuracy above. Now we are going to encode the rest of the variables too. 
# We define the function transform_data which encodes the following variables:
# * **Age:** break the age variable into age groups (baby, child, etc) and one-hot-encode the resulting categorical variable
# * **Cabin**: take the first letter of the cabin, wich denotes the class of  the cabin, and one-hot-encode that as well
# * **Pclass**: we will encode pclass too. While it might appear as numerical at a first glance it's categorical actually since it has discrete values with an assigned meaning and it's not a numerical quantity

# In[8]:


def transform_data(df, labeled_data=True):
    df.Age = df.Age.fillna(-0.5)
    df.Fare = df.Fare.fillna(0)
    df.Cabin = df.Cabin.fillna('N')

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    df.Age = pd.cut(df.Age, bins, labels=group_names)

    df.Sex = df.Sex.apply(lambda x: 0 if x == 'male' else 1)
    df.Cabin = df.Cabin.apply(lambda x: x[0])

    cat_vars=['Cabin', 'Age', 'Pclass'] # 'Fare']
    for v in cat_vars:
        dummies = pd.get_dummies(df[v], prefix=v)
        df = df.join(dummies)
    df.drop(cat_vars, axis=1, inplace=True)
    
    return df


# Train and test our model again, see if the feature engineering we did above improved our score

# In[9]:


train_df = pd.read_csv('../input/train.csv')

train_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
train_df = transform_data(train_df)

y = train_df['Survived']
X = train_df.drop('Survived', axis=1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)

lr_model = LogisticRegression()
lr_model.fit(Xtrain, ytrain)
predictions = lr_model.predict(Xtest)

print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))
print('Confusion matrix:')
print(confusion_matrix(predictions, ytest, labels=[0, 1]))


# Now we run our model again and make our 2nd submission, hopefully with a better score

# In[10]:


train_df = pd.read_csv('../input/train.csv')
final_df = pd.read_csv('../input/test.csv')
test_df = pd.read_csv('../input/test.csv')

y = train_df['Survived']
train_df.drop('Survived', axis=1, inplace=True)

df = pd.concat([train_df, test_df]).reset_index()
df.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

df = transform_data(df)
train_df = df[:len(train_df)]
final_df = df[len(train_df):]

pid = final_df['PassengerId']
Xfinal =  final_df.drop('PassengerId', axis=1)

X = train_df.drop('PassengerId', axis=1)

lr_model.fit(X, y)

final_predictions = lr_model.predict(Xfinal)

if not os.path.isdir('./output'):
    os.mkdir('output')
results = pd.DataFrame({'PassengerId': pid, 'Survived': final_predictions})
results.to_csv('./output/submission1b.csv', index=False)


# # 2. Gradient Boosting Decision Trees with LightGBM

# In[11]:


import lightgbm as lgb


# Since we extracted as much value as possible from the simple logistic regression model, we are now moving to a highest performing one, LightGBM.
# 
# First, we will need to do the same transformations to the data as before

# In[12]:


train_df = pd.read_csv('../input/train.csv')
final_df = pd.read_csv('../input/test.csv')
test_df = pd.read_csv('../input/test.csv')

y = train_df['Survived']
train_df.drop('Survived', axis=1, inplace=True)

df = pd.concat([train_df, test_df]).reset_index()
df.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

df = transform_data(df)
train_df = df[:len(train_df)]
final_df = df[len(train_df):]

pid = final_df['PassengerId']
Xfinal =  final_df.drop('PassengerId', axis=1)

X = train_df.drop('PassengerId', axis=1)


# Create a basic lightgbm model and check its accuracy

# In[13]:


lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    objective='binary',
    n_jobs=4,
)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)

lgb_model.fit(Xtrain, ytrain)
predictions = lgb_model.predict(Xtest)

print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))
print('Confusion matrix:')
print(confusion_matrix(predictions, ytest, labels=[0, 1]))


# We can now fine tune the model by changing its parameters. LightGBM has close to 100 parameters that  we can fine tune, but in this example some of the basic  are shown. 
# Each time we change a parameter we cross-validate the model by running it multiple times over the training dataset and calculating the mean accuracy. When we are satisfied with our settings we can move on and test it on the testing dataset.
# 

# In[14]:


lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    num_leaves=25,
    objective='binary',
    n_jobs=4,
)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)

predictions = cross_val_predict(lgb_model, Xtrain, ytrain, cv=10)

print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytrain)))
print('Confusion matrix:')
print(confusion_matrix(predictions, ytrain, labels=[0, 1]))


# We shall continue testing our model on the testing dataset, and if we are satisfied with the accuracy we can move on to calculate the final submission results. Otherwise we can return to the previous step and do more fine tuning. If we find that the accuracy on the test dataset is significantly worse than the validation then we are probably **overfitting**.

# In[15]:


lgb_model.fit(Xtrain, ytrain)
predictions = lgb_model.predict(Xtest)

print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))
print('Confusion matrix:')
print(confusion_matrix(predictions, ytest, labels=[0, 1]))


# Now, let's create a submission file

# In[16]:


lgb_model.fit(X, y)

final_predictions = lgb_model.predict(Xfinal)

if not os.path.isdir('./output'):
    os.mkdir('output')
results = pd.DataFrame({'PassengerId': pid, 'Survived': final_predictions})
print(results.head(20))
results.to_csv('./output/submission2a.csv', index=False)


# In[ ]:


X.head()

