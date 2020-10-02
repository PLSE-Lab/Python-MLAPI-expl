#!/usr/bin/env python
# coding: utf-8

# # Welcome to the legendary Titanic ML competition
# This notebook is a starter code for all beginners and easy to understand. We will give an introduction to analysis and feature engineering.<br> 
# Therefore we focus on
# * a simple analysis of the data,
# * create new features,
# * encoding and
# * scale data.
# 
# We use categorical feature encoding techniques, compare <br>
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb <br>
# The data is descriped here: https://www.kaggle.com/c/titanic/data <br>
# We label the **necessary** operations and the operations for **advanced** feature engeneering. So for the first run you can skip the advanced feature engeneering.
# 
# Finally we define a simple classification model without optimization of the parameters.

# # Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import os


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# # Input path

# In[ ]:


path_in = '../input/titanic/'


# In[ ]:


os.listdir(path_in)


# # Load data

# In[ ]:


train_data = pd.read_csv(path_in+'train.csv', index_col=0)
test_data = pd.read_csv(path_in+'test.csv', index_col=0)
samp_subm = pd.read_csv(path_in+'gender_submission.csv', index_col=0)


# # Plot function

# In[ ]:


def plot_bar_compare(train, test, name, rot=False):
    """ Compare the distribution between train and test data """
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    
    train_label = train[name].value_counts().sort_index()
    dict_train = dict(zip(train_label.keys(), ((100*(train_label)/len(train.index)).tolist())))
    train_names = list(dict_train.keys())
    train_values = list(dict_train.values())
    
    test_label = test[name].value_counts().sort_index()
    dict_test = dict(zip(test_label.keys(), ((100*(test_label)/len(test.index)).tolist())))
    test_names = list(dict_test.keys())
    test_values = list(dict_test.values())
    
    axs[0].bar(train_names, train_values, color='yellowgreen')
    axs[1].bar(test_names, test_values, color='sandybrown')
    axs[0].grid()
    axs[1].grid()
    axs[0].set_title('Train data')
    axs[1].set_title('Test data')
    axs[0].set_ylabel('%')
    if(rot==True):
        axs[0].set_xticklabels(train_names, rotation=45)
        axs[1].set_xticklabels(test_names, rotation=45)
    plt.show()


# # Overview
# The titanic data set is small. In total we have only 1309 samples.

# In[ ]:


print('number train samples: ', len(train_data.index))
print('number test samples: ', len(test_data.index))


# In[ ]:


train_data.head()


# # Handle missing values (necessary)

# In[ ]:


cols_with_missing_train = [col for col in train_data.columns if train_data[col].isnull().any()]
cols_with_missing_test = [col for col in test_data.columns if test_data[col].isnull().any()]


# In[ ]:


print('train columns with missing data:', cols_with_missing_train)
print('test columns with missing data:', cols_with_missing_test)


# ## Label rows with missing values (advanced)
# We can create new features and label the rows with missing values. This we have to do before handling missing values. For fit and prediction the train and test data must have the same number of columns. So we have to merge the missing columns.

# In[ ]:


cols_with_missing_data = list(set(cols_with_missing_train + cols_with_missing_test))


# In[ ]:


for col in cols_with_missing_data:
    train_data[col + '_was_missing'] = train_data[col].isnull()
    test_data[col + '_was_missing'] = test_data[col].isnull()


# ## Feature Age
# This is a numerical feature. There are 177 missing values in the train and 86 missing values in the testd data. There are several techniques to fill the missing data, i.e. set them to zero oder the mean value.

# In[ ]:


print('train missing values:', train_data['Age'].isna().sum())
print('test missing values:', test_data['Age'].isna().sum())


# In[ ]:


train_data['Age'] = train_data['Age'].fillna(0, inplace=False)
test_data['Age'] = test_data['Age'].fillna(0, inplace=False)


# ## Feature Cabin
# This is a categorical feature. 
# There are 687 missing values in the train and 327 missing values in the testd data. We fill the missing values with the new category *Unkown*.

# In[ ]:


print('train missing values:', train_data['Cabin'].isna().sum())
print('test missing values:', test_data['Cabin'].isna().sum())


# In[ ]:


train_data['Cabin'] = train_data['Cabin'].fillna('Unknown', inplace=False)
test_data['Cabin'] = test_data['Cabin'].fillna('Unknown', inplace=False)


# ## Feature Embarked
# The Port of Embarkation is a categorical feature. There are 2 missing values in the train data set. We fill the missing values with the new category *Unkown*.

# In[ ]:


print('train missing values:', train_data['Embarked'].isna().sum())


# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna('Unknown', inplace=False)


# ## Feature Fare
# The passenger fare is a numerical feature. There is one missing value in the test data. We fill this with the mean value.

# In[ ]:


print('test missing values:', test_data['Fare'].isna().sum())


# In[ ]:


mean = test_data['Fare'].mean()
test_data['Fare'] = test_data['Fare'].fillna(mean, inplace=False)


# # Create new features (advanced)
# Based on the given features we are able to create new features.
# ## Family Size
# We combine the both features *SibSp* (number of  of siblings / spouses aboard the Titanic) and *Parch* (number of parents / children aboard the Titanic) to the *FamilySize*.

# In[ ]:


train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1


# ## Familiy Group
# With the new label *FamilySize* we can create the family group based on the number of members.

# In[ ]:


def family_group(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0


# In[ ]:


train_data['FamilyGroup'] = train_data['FamilySize'].apply(family_group)
test_data['FamilyGroup'] = test_data['FamilySize'].apply(family_group)


# ## Ticket group
# We see there are multiple ticket labels. So we can define groups of them. For this we have to sum up the train and test data.

# In[ ]:


train_data['Ticket'].value_counts()


# In[ ]:


ticket_group = dict((train_data['Ticket'].append(test_data['Ticket'])).value_counts())
train_data['TicketGroup'] = train_data['Ticket'].apply(lambda x: ticket_group[x])
test_data['TicketGroup'] = test_data['Ticket'].apply(lambda x: ticket_group[x])


# ## Title of a passenger
# From the feature *Name* we can extract the title of the passengers. Further we want to sum up some title to one title, i.e. instead of *Mme*, *Ms* and *Mrs* we only use *Mrs*. We will see that the distribution is similar between train and test.

# In[ ]:


train_data['Title'] = train_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
test_data['Title'] = test_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
title_dict = {}
title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
train_data['Title'] = train_data['Title'].map(title_dict)
test_data['Title'] = test_data['Title'].map(title_dict)


# In[ ]:


plot_bar_compare(train_data, test_data, 'Title', rot=True)


# ## Age
# We can group the age for simplification. For this we use 9 groups (class 0 is for the missing values). We can see there are significant 
# different distributions for class 3 and 4.

# In[ ]:


def age_group(s):
    if (s > 0) & (s < 10):
        return 1
    elif (s > 10) & (s <= 10):
        return 2
    elif (s > 20) & (s <= 30):
        return 3
    elif (s > 30) & (s <= 40):
        return 4
    elif (s > 40) & (s <= 50):
        return 5
    elif (s > 50) & (s <= 60):
        return 6
    elif (s > 60) & (s <= 70):
        return 7
    elif (s > 70) & (s <= 80):
        return 8


# In[ ]:


train_data['AgeGroup'] = train_data['Age'].apply(age_group)
test_data['AgeGroup'] = test_data['Age'].apply(age_group)


# In[ ]:


plot_bar_compare(train_data, test_data, 'AgeGroup', rot=False)


# ## Fare group

# In[ ]:


def fare_group(s):
    if (s <= 8):
        return 1
    elif (s > 8) & (s <= 15):
        return 2
    elif (s > 15) & (s <= 31):
        return 3
    elif s > 31:
        return 4


# In[ ]:


train_data['FareGroup'] = train_data['Fare'].apply(fare_group)
test_data['FareGroup'] = test_data['Fare'].apply(fare_group)


# # Compare train and test data (advanced)
# In the previous picture we can see that there significant differences for the *Age*. What is with other features?
# ## Sex
# The distributions are similar.

# In[ ]:


plot_bar_compare(train_data, test_data, 'Sex', rot=False)


# ## Embarked
# There are differences between train and test.

# In[ ]:


plot_bar_compare(train_data, test_data, 'Embarked', rot=False)


# ## FamilySize
# The distributions are similar.

# In[ ]:


plot_bar_compare(train_data, test_data, 'FamilySize', rot=False)


# ## TicketGroup
# The distributions are similar.

# In[ ]:


plot_bar_compare(train_data, test_data, 'TicketGroup', rot=False)


# ## FareGroup
# The distributions are similar.

# In[ ]:


plot_bar_compare(train_data, test_data, 'FareGroup', rot=False)


# In[ ]:


train_data['Fare'].describe()


# # Encoding data (necessary)
# In this section we show how to deal with encoding techniques. For this we recommend another competition:
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb <br>
# There are some classes of features we have to encode with different techniques: <br>
# 1) categorical features: Sex, Cabin, Embarked and Title. <br>
# 2) binary features: Age_was_missing, Cabin_was_missing,	Embarked_was_missing <br>
# 3) ordinal features: Pclass <br>
# We leave out the features Name and Ticket because we don't want to use for prediction.

# ## Feature Cabin
# Bevor starting with encoding we want so simplify the values of the feature *Cabin*. We only want to use the first character. The cabine group *T* is only included in the train data.

# In[ ]:


train_data['Cabin'] = train_data['Cabin'].str[0]
test_data['Cabin'] = test_data['Cabin'].str[0]


# In[ ]:


plot_bar_compare(train_data, test_data, 'Cabin', rot=False)


# ## Categorical features
# We use the simple LabelEncoder. 

# In[ ]:


features_cat = ['Sex', 'Cabin', 'Embarked', 'Title']
le = LabelEncoder()
for col in features_cat:
    le.fit(train_data[col])
    train_data[col] = le.transform(train_data[col])
    test_data[col] = le.transform(test_data[col])


# ## Ordinal features
# For the feature *Pclass* we use the one-hot-encoding.

# In[ ]:


plot_bar_compare(train_data, test_data, 'Pclass', rot=False)


# In[ ]:


train_data_dummy = pd.get_dummies(train_data['Pclass'], prefix = 'Pclass')
train_data = pd.concat([train_data, train_data_dummy], axis=1)
del train_data['Pclass']
test_data_dummy = pd.get_dummies(test_data['Pclass'], prefix = 'Pclass')
test_data = pd.concat([test_data, test_data_dummy], axis=1)
del test_data['Pclass']


# # Select features
# We don't want to use all features, i.e. *Name*. And the column *Survived* is the target we have to predict.

# In[ ]:


no_features = ['Survived', 'Name', 'Ticket', 'Age', 'Fare']


# # Define X_train, y_train and X_test

# In[ ]:


X_train = train_data[train_data.columns.difference(no_features)].copy(deep=False)
y_train = train_data['Survived']
X_test = test_data[test_data.columns.difference(no_features)].copy(deep=False)


# # Scale date
# To avoid numercial effects between small and large numbers we scale the data.

# In[ ]:


scale_features = ['AgeGroup', 'FareGroup', 'Cabin', 'Embarked', 'FamilySize', 'FamilyGroup', 'TicketGroup', 'Title']


# In[ ]:


mean = X_train[scale_features].mean(axis=0)
X_train[scale_features] = X_train[scale_features].astype('float32')
X_train[scale_features] -= X_train[scale_features].mean(axis=0)
std = X_train[scale_features].std(axis=0)
X_train[scale_features] /= X_train[scale_features].std(axis=0)
X_test[scale_features] = X_test[scale_features].astype('float32')
X_test[scale_features] -= mean
X_test[scale_features] /= std


# # Split train and val data
# To calibrate the model we split the train data into train and val data.

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.001, random_state=2019)


# # Define the model
# We use the XGB Classifier

# In[ ]:


model = XGBClassifier(objective ='binary:logistic',
                      learning_rate = 0.01,
                      max_depth = 15,
                      n_estimators = 100,
                      random_state=2019)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


preds_val = model.predict(X_val)


# # Calc categorization accuracy on val data

# In[ ]:


accuracy_score(y_val, preds_val)


# # Predict test data

# In[ ]:


y_test = model.predict(X_test)


# # Write output for submission

# In[ ]:


output = pd.DataFrame({'PassengerId': samp_subm.index,
                       'Survived': y_test})
output.to_csv('submission.csv', index=False)


# In[ ]:


output

