#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Work on analysis and cleaning

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


train_df['Survived'].unique()


# In[ ]:


train_df['Survived'].value_counts(normalize=True) * 100


# In[ ]:


survival_perc = train_df['Survived'].value_counts(normalize=True) * 100
survival_perc.plot(kind='bar')

plt.ylabel('Percentage of Survival')
plt.ylim(0, 65)
plt.show()


# In[ ]:


train_df['Pclass'].unique()


# In[ ]:


train_df['Sex'].unique()


# In[ ]:


def name_to_title(name) :
    if "Mr." in name:
        return "Mr"
    elif "Mrs." in name:
        return "Mrs"
    elif "Master" in name:
        return "Master"
    elif "Miss" in name:
        return "Miss"
    else :
        return "Highness"

name_df = train_df[['Name', 'Survived']]
name_df['Title'] = name_df['Name'].apply(name_to_title)
name_df = name_df.drop('Name', axis=1)

name_df.groupby(['Title']).mean().plot(kind='bar')
plt.show()


# In[ ]:


train_df['Age'].describe()


# In[ ]:


train_df['Age'].isnull().sum()


# In[ ]:


print(train_df['Age'].mean())

plt.hist(train_df['Age'])
plt.show()


# In[ ]:


name_df = train_df[['Name', 'Survived', 'Age']]
name_df['Title'] = name_df['Name'].apply(name_to_title)

mean_adult_male_age = round(name_df[name_df['Title'] == 'Mr']['Age'].mean(), 2)
mean_adult_female_age = round(name_df[name_df['Title'] == 'Mrs']['Age'].mean(), 2)
mean_young_female_age = round(name_df[name_df['Title'] == 'Miss']['Age'].mean(), 2)
mean_young_male_age = round(name_df[name_df['Title'] == 'Master']['Age'].mean(), 2)

# TODO: row mapper for null values
def missing_age_mapper(row) :
    if np.isnan(row['Age']):
        if row['Title'] == 'Mr':
            return (mean_adult_male_age)
        elif row['Title'] == 'Mrs':
            return (mean_adult_female_age)
        elif row['Title'] == 'Miss':
            return (mean_young_female_age)
        else:
            return (mean_young_male_age)
    else:
        return row['Age']
    
name_df['Age'] = name_df.apply(missing_age_mapper, axis=1)

plt.hist(train_df['Age'])
plt.show()


# In[ ]:


test_age = train_df[['Survived', 'Age']]
dead_age = test_age[test_age['Survived'] == 0]
survived_age = test_age[test_age['Survived'] == 1]

plt.subplot(1, 2, 1)
plt.hist(dead_age['Age'])

plt.subplot(1, 2, 2)
plt.hist(survived_age['Age'])

plt.show()


# In[ ]:


test_df = train_df[['Survived', 'Embarked']]
grouped_by_sx_em = test_df.groupby(['Embarked']).mean()

grouped_by_sx_em


# In[ ]:


grouped_by_sx_em.plot(kind='bar')
plt.show()


# In[ ]:


mean_srv_pclass = train_df[['Survived', 'Pclass']].groupby('Pclass').mean()
mean_srv_pclass.plot(kind='bar')


# In[ ]:


train_df[['Survived', 'Cabin']].info()


# In[ ]:


train_df[['Survived', 'Ticket']].info()


# In[ ]:


train_df[['Survived', 'Ticket']].head()


# ### Observations
# 
# - Gender and Age has a high correlation to the survival rate.
# - We can extract title from Names column
# - Ticket and Cabin columns do not have any correlation to survival rate
# - We can fill in empty Age data from mean of each title
# - We can improve performance by changing data types of some columns

# ### Processing

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


def name_to_title(name) :
    if "Mr." in name:
        return 1
    elif "Mrs." in name:
        return 2
    elif "Master" in name:
        return 3
    elif "Miss" in name:
        return 4
    else :
        return 5

train_df['Title'] = train_df['Name'].apply(name_to_title)
test_df['Title'] = test_df['Name'].apply(name_to_title)

mean_adult_male_age = round(name_df[train_df['Title'] == 1]['Age'].mean(), 2)
mean_adult_female_age = round(name_df[train_df['Title'] == 2]['Age'].mean(), 2)
mean_young_female_age = round(name_df[train_df['Title'] == 3]['Age'].mean(), 2)
mean_young_male_age = round(name_df[train_df['Title'] == 4]['Age'].mean(), 2)

# TODO: row mapper for null values
def missing_age_mapper(row) :
    if np.isnan(row['Age']):
        if row['Title'] == 1:
            return (mean_adult_male_age)
        elif row['Title'] == 2:
            return (mean_adult_female_age)
        elif row['Title'] == 3:
            return (mean_young_female_age)
        else:
            return (mean_young_male_age)
    else:
        return row['Age']
    
train_df['Age'] = train_df.apply(missing_age_mapper, axis=1)
test_df['Age'] = test_df.apply(missing_age_mapper, axis=1)


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna(method='ffill')

train_df.head()


# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch'], axis=1)
test_df = test_df.drop(['Name', 'SibSp', 'Parch'], axis=1)


# In[ ]:


# train_df['Pclass'].unique()


# In[ ]:


train_df.info()


# In[ ]:


test_df = test_df.fillna(method='ffill')


# In[ ]:





# In[ ]:


test_df.info()


# In[ ]:


train_df['Title'].unique()


# ### Prediction

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf

# make TensorFlow less verbose
tf.logging.set_verbosity(tf.logging.ERROR)

# read the dataset
train_data = train_df
test_data = test_df

# drop unused columns


# In[ ]:


# sample 80% for train data
train_set = train_data.sample(frac=0.8, replace=False, random_state=42)
# the other 20% is reserved for cross validation
cv_set = train_data.loc[ set(train_data.index) - set(train_set.index)]

# define features
sex_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Sex', ['female','male']
)

pclass_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Pclass', [1, 2, 3]
)

embarked_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Embarked', ['S', 'C', 'Q']
)

title_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Title', [1, 2, 4, 3, 5]
)


feature_columns = [ 
    tf.feature_column.indicator_column(sex_feature),
    tf.feature_column.indicator_column(pclass_feature),
    tf.feature_column.indicator_column(embarked_feature),
    tf.feature_column.indicator_column(title_feature)
]

n_batches = 1
estimator = tf.estimator.BoostedTreesClassifier(feature_columns,n_batches_per_layer=n_batches)

# train input function
train_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=train_set.drop('Survived', axis=1),
      y=train_set.Survived,
      num_epochs=None, # for training, use as many epochs as necessary
      shuffle=True,
      target_column='target',
)

cv_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=cv_set.drop('Survived', axis=1),
      y=cv_set.Survived,
      num_epochs=1, # only to score
      shuffle=False
)


# In[ ]:




estimator.train(input_fn=train_input_fn, steps=50)

scores = estimator.evaluate(input_fn=cv_input_fn)
print("\nTest Accuracy: {0:f}\n".format(scores['accuracy']))


# In[ ]:




test_input_fn = tf.estimator.inputs.pandas_input_fn(
      x=test_data,
      num_epochs=1, # only to predict
      shuffle=False 
)

predictions = list(estimator.predict(input_fn=test_input_fn))
predicted_classes = [prediction['class_ids'][0] for prediction in predictions]
evaluation = test_data['PassengerId'].copy().to_frame()
evaluation["Survived"] = predicted_classes
evaluation.to_csv("evaluation_submission.csv", index=False)

evaluation.head()

