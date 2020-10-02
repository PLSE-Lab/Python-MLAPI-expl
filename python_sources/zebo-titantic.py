#!/usr/bin/env python
# coding: utf-8

# > # 0. Setup
# >> ## 0.1. Libraries
# * NumPy and pandas are used for exploratory data analysis in order to summarize the main characteristics of the data
# * NumPy and pandas are also used for feature engineering which will come in handy later in machine learning
# * matplotlib is used for visualization in order to assist data analysis
# * Scikit-learn is used for the machine learning algorithms, train/test split, model evaluation, etc.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn import feature_selection
from sklearn import metrics
from sklearn import linear_model, ensemble, gaussian_process
from xgboost import XGBClassifier


# >> ## 0.2. Loading the data set
# After loading the train and test sets to the memory, copying them recursively with the `copy()` function because we don't want changes to be reflected to the original data frame. After that, assigning a name attribute for data frames for later use
# 

# In[ ]:


df_train_orig = pd.read_csv('../input/train.csv')
df_test_orig = pd.read_csv('../input/test.csv')

df_train = df_train_orig.copy(deep=True)
df_train.name = 'Training set'
df_test = df_test_orig.copy(deep=True)
df_test.name = 'Test set'


# > # 1. Data Analysis
# >> ## 1.1. Overview
# * We use `info()` to get an overview of the types of the features
# * Using `sample(10)` to get random 10 rows from the training set

# In[ ]:


print(df_train_orig.info())
df_train_orig.sample(10)


# >> ## 1.2. Fixing null values
# As seen from the random sample, the Cabin and Age columns have null values. They have to be managed but let's see which columns also have null values and how many. This function below outputs the sum of null values in all columns in both training and test set.

# In[ ]:


def show_nulls(df):
    print('{} columns with null values '.format(df.name))
    print(df.isnull().sum())
    print("\n")
    
for df in [df_train, df_test]:
    show_nulls(df)


# >> * Training set have null values in Age, Cabin and Embarked columns
# * Test set have null values in Age, Fare and Embarked columns.
# 
# >> The percentage of null values in Age, Embarked and Fare columns are relatively smaller compared to the length of the sets, but more than 80% of the Cabin column are null values in both training and test tests. In this case, we fill the null values of Age column with median, Embarked column with mode since it is categorical and Fare column with median.
# 
# >> Since the large portion of the Cabin column is missing and if we fill the null values, it will dramatically affect the training accuracy. The model will probably fail on test set. That's why we are dropping Cabin column along with PassengerId and Ticket in the training set. PassengerId and Ticket columns are dropped because they are unique values and they don't have any effect on the output.

# In[ ]:


for df in [df_train, df_test]:    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
df_train.drop(['PassengerId','Cabin', 'Ticket'], axis=1, inplace=True)


# >> Checking the null values again and we can see that there are no null values left in the training set since we dropped the Cabin column

# In[ ]:


for df in [df_train, df_test]:
    show_nulls(df)


# >> ## 1.3. Checking the distribution of data
# Y_train is not equally distributed, but the gap is not that big, so the bias is not significant. We don't need to balance the distribution in this case.

# In[ ]:


df_survive = df_train_orig['Survived'].value_counts()
print(df_survive)
ax = df_survive.plot.bar()
ax.set_xticklabels(('Not Survived', 'Survived'))


# >> ## 1.4. Feature Engineering
# * Family_Members is created by adding SibSp, Parch and 1. Since we know that SibSp is siblings and spouse, and Parch is parents and children, we can add those columns to find the count of family members of the person. Finally, adding 1 is the person himself or herself.
# * Is_Alone column is based on the number of Family_Members. If Family_Members' value is more than 1, Is_Alone is set to 0, otherwise it is set to 1
# * Title column is created by extracting prefix before the Name column

# In[ ]:


for df in [df_train, df_test]:    
    df['Family_Members'] = df['SibSp'] + df['Parch'] + 1
    
    df['Is_Alone'] = 1
    df['Is_Alone'].loc[df['Family_Members'] > 1] = 0
    
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0] 

df_train.sample(10)


# >> Since the Title column is categorical, we can group up some values to a big one. Titles like Master and Dr might have a higher priority at the evacuation, so this feature might be worth exploring. We are going group Titles that are coming after Dr to Other because their titles are not as significant as others I think.

# In[ ]:


df_train['Title'].value_counts()


# >> Titles that are less than 7, are grouped into Other, so we keep the number of doctors.

# In[ ]:


train_title_names = (df_train['Title'].value_counts() < 10)
df_train['Title'] = df_train['Title'].apply(lambda x: 'Other' if train_title_names.loc[x] == True else x)

df_train['Title'].value_counts()


# In[ ]:


test_title_names = (df_test['Title'].value_counts() < 10)
df_test['Title'] = df_test['Title'].apply(lambda x: 'Other' if test_title_names.loc[x] == True else x)

df_test['Title'].value_counts()


# >> ## 1.5. Categorical to dummy
# Categorical data are transformed to numerical data with the `LabelEncoder()` from scikit-learn. It basically labels the categories from 0 to n.

# In[ ]:


le = LabelEncoder()
for df in [df_train, df_test]:    
    df['Sex_Label'] = le.fit_transform(df['Sex'])
    df['Embarked_Label'] = le.fit_transform(df['Embarked'])
    df['Title_Label'] = le.fit_transform(df['Title'])
    
df_train.head()


# >> The labeled columns are converted to one-hot encoding with `get_dummies()` function. The given columns are converted to one-hot encoding if they are labels of the categorical data.

# In[ ]:


X_cols = ['Sex', 'Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'Family_Members', 'Is_Alone']

df_train_dummy = pd.get_dummies(df_train[X_cols])
df_test_dummy = pd.get_dummies(df_test[X_cols])
df_train_dummy['Survived'] = df_train['Survived']

df_train_dummy.head()


# >> ## 1.6. Separating X and Y
# * Pclass, SibSp, Parch, Age, Fare, Family_Members, Is_Alone, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S, Title_Master, Title_Miss, Title_Mr, Title_Mrs, Title_Other is  X_train (Training Input)
# * Survived is Y_train (Training Output)

# In[ ]:


X_train = df_train_dummy.drop(['Survived'], axis=1)
Y_train = df_train_dummy['Survived']


# > # 2. Machine Learning
# >> ## 2.1 Models
# We are going to try several types of machine learning algorithms with different parameters in this part. First storing the models in a list and fitting them.

# In[ ]:


seed = 0

models = [ensemble.RandomForestClassifier(n_estimators=65, min_impurity_decrease=0.1, random_state=seed),
          ensemble.AdaBoostClassifier(n_estimators=11, algorithm='SAMME.R'),
          ensemble.GradientBoostingClassifier(loss='exponential', learning_rate=0.01, n_estimators=100, criterion='friedman_mse', max_depth=4),
         linear_model.LogisticRegressionCV(cv=3, penalty='l2', solver='newton-cg'),
         XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)]

fits = [model.fit(X_train, Y_train) for model in models]

fits


# >> ## 2.2 Predictions
# Each of the predictions predicted by the models are stored in Y_hats. It's dictionary of model name:predictions pairs.

# In[ ]:


Y_hats = {model.__class__.__name__: model.predict(X_train) for model in models}


# > # 3. Evaluation
# >> ## 3.1 Cross-validation
# We are scoring the model with cross-validation. `ShuffleSplit()` is a random permutation cross-validator which yields indices to split data into training and test sets. Evaluating the models with the cross-validation by their train_score and test_score. 

# In[ ]:


cv_split = ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=seed)
cv_split


# In[ ]:


for model in models:
    cv_results = cross_validate(model, X_train, Y_train, cv=cv_split)
    print(model.__class__.__name__)
    print(cv_results)


# >> ## 3.2 F1-score
# Evaluating the models with F1-score.

# In[ ]:


for model, Y_hat in Y_hats.items():
    print(model)
    print(metrics.classification_report(Y_train, Y_hat, target_names=['Not Survived', 'Survived']))
    print('\n')


# > # 4. Submission
# We can conclude that the highest performance is achieved by the XGBClassifier model, so we are using it in the final submission..

# In[ ]:


submission_model = models[4]
submission_model


# In[ ]:


submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = df_test_orig['PassengerId']
submission_df['Survived'] = submission_model.predict(df_test_dummy)


# In[ ]:


submission_df.head(10)


# In[ ]:


submission_df.to_csv('submissions.csv', header=True, index=False)

