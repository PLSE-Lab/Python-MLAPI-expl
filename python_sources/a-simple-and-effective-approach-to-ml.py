#!/usr/bin/env python
# coding: utf-8

# This notebook is a simple yet an effective approach to Machine Learning for the Titanic dataset. After having gone through multiple notebooks and approaches, I have come to realize that the simpler the approach, the better the results for this dataset. The more complexity in feature engineering I applied, the more the model overfit, giving poor results on the test dataset. 
# 
# This notebook does not explain ML concepts, but concludes learnings and findings wherever necessary. It follows a very straight forward step by step ML approach with simple python commands.
# 
# It follows the following steps:
# * Basic exploratory data analysis
# * Analyzing features with the target column
# * Feature engineering
# * Treating missing values
# * Feature engineering continued
# * Dropping redundant features
# * Creating dummies for all features
# * Separating target and features
# * Predictive modeling
# * Hyperparameter tuning using grid search
# * Comparing accuracy scores
# 
# If you have any questions or feedback, please let me know in the comment section. 

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style('whitegrid')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')


# ### Basic exploratory data analysis

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=['O'])


# In[ ]:


# Understanding the target class is very important

sns.countplot('Survived', data=df)


# In[ ]:


100.0*df['Survived'].value_counts() / len(df)


# - Training data contains 891 samples (40%) compared to 2205 total passengers on board
# - 61.6% of the people did not survive
# - 38.38% of the people survived comapred to the 32% survival rate of the complete dataset

# In[ ]:


df.corr()['Survived']


# ### Analysing features with the target column

# #### Pclass, Sex, SibSp, Parch

# In[ ]:


df[['Pclass','Survived']].groupby('Pclass', as_index=False).mean()


# In[ ]:


df[['Sex', 'Survived']].groupby('Sex', as_index=False).mean()


# In[ ]:


df[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values('Survived', ascending=False)


# In[ ]:


df[['Parch','Survived']].groupby('Parch', as_index=False).mean().sort_values('Survived', ascending=False)


# - 69% of Pclss=1 passangers survived
# - 74% of females survived
# - Passangers with lesser SibSp have a higher survival rate
# - SibSp and Parch have zero correlation for certain values

# #### Age

# In[ ]:


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=15)


# - Infants (age<5) have a high survival rate
# - Oldest peole (age=80) all survived
# - Most passangers in 15-25 age range, highest mortality rate in that range

# #### Pclass, Survived Vs. Age

# In[ ]:


g = sns.FacetGrid(df, col='Survived', row='Pclass')
g.map(plt.hist, 'Age')


# - Infants with Pclass 1 and 2 mostly survived
# - Most adults in Pclass 3 did not surivive
# - Most passangers in Pclass 1 survived

# #### Embarked

# In[ ]:


df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean().sort_values('Survived', ascending=False)


# In[ ]:


100.0*df['Embarked'].value_counts() / len(df)


# - 72% of the passengers on board embarked from port S
# - Port S also has the highest number of survivors, 55%

# ### Feature Engineering

# Make sure to perform actions simultaneously on both, the train and test dataset

# #### Extracting Title from the Name feature

# In[ ]:


df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.')


# In[ ]:


df.head()


# In[ ]:


df['Title'].value_counts()


# #### Merging Titles 
# Comparing Titles with the Sex feature to figure out how rarely used titles can be merged with other titles

# In[ ]:


pd.crosstab(df['Title'], df['Sex'])


# In[ ]:


replace_titles = ['Capt','Col','Countess','Don','Jonkheer','Lady','Major','Dr','Rev','Sir']


# In[ ]:


df['Title'] = df['Title'].replace(replace_titles, 'other')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')


# In[ ]:


df[['Title','Survived']].groupby('Title').mean().sort_values('Survived', ascending=False)


# In[ ]:


test_df['Title'] = test_df['Title'].replace(replace_titles, 'other')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')


# In[ ]:


pd.crosstab(test_df['Title'], test_df['Sex'])


# Oh, test set has another unique title called 'dona', merging that with other

# In[ ]:


test_df['Title'] = test_df['Title'].replace('Dona', 'other')


# ### Treating Mising Values before performing further Feature Engineering

# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


test_df.isnull().sum().sort_values(ascending=False)


# #### Filling missing values for age based on mean age per Title

# In[ ]:


print('Number of age entries missing for title Miss:', df[df['Title'] == 'Miss']['Age'].isnull().sum())
print('Number of age entries missing for title Mr:', df[df['Title'] == 'Mr']['Age'].isnull().sum())
print('Number of age entries missing for title Mrs:', df[df['Title'] == 'Mrs']['Age'].isnull().sum())
print('Number of age entries missing for title other:', df[df['Title'] == 'other']['Age'].isnull().sum())
print('Number of age entries missing for title Master:', df[df['Title'] == 'Master']['Age'].isnull().sum())


# In[ ]:


print('Mean age for title Miss:', df[df['Title'] == 'Miss']['Age'].mean())
print('Mean age for title Mr:', df[df['Title'] == 'Mr']['Age'].mean())
print('Mean age for title Mrs:', df[df['Title'] == 'Mrs']['Age'].mean())
print('Mean age for title other:', df[df['Title'] == 'other']['Age'].mean())
print('Mean age for title Master:', df[df['Title'] == 'Master']['Age'].mean())


# In[ ]:


df.loc[(df['Title']== 'Miss') & (df['Age'].isnull()), 'Age'] = 22
df.loc[(df['Title']== 'Mr') & (df['Age'].isnull()), 'Age'] = 32
df.loc[(df['Title']== 'Mrs') & (df['Age'].isnull()), 'Age'] = 36
df.loc[(df['Title']== 'other') & (df['Age'].isnull()), 'Age'] = 46
df.loc[(df['Title']== 'Master') & (df['Age'].isnull()), 'Age'] = 5


# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


# Repeating the steps for test set

print('Number of age entries missing for title Miss:', test_df[test_df['Title'] == 'Miss']['Age'].isnull().sum())
print('Number of age entries missing for title Mr:', test_df[test_df['Title'] == 'Mr']['Age'].isnull().sum())
print('Number of age entries missing for title Mrs:', test_df[test_df['Title'] == 'Mrs']['Age'].isnull().sum())
print('Number of age entries missing for title other:', test_df[test_df['Title'] == 'other']['Age'].isnull().sum())
print('Number of age entries missing for title Master:', test_df[test_df['Title'] == 'Master']['Age'].isnull().sum())


# In[ ]:


print('Mean age for title Miss:', test_df[test_df['Title'] == 'Miss']['Age'].mean())
print('Mean age for title Mr:', test_df[test_df['Title'] == 'Mr']['Age'].mean())
print('Mean age for title Mrs:', test_df[test_df['Title'] == 'Mrs']['Age'].mean())
print('Mean age for title other:', test_df[test_df['Title'] == 'other']['Age'].mean())
print('Mean age for title Master:', test_df[test_df['Title'] == 'Master']['Age'].mean())


# In[ ]:


test_df.loc[(test_df['Title']== 'Miss') & (test_df['Age'].isnull()), 'Age'] = 22
test_df.loc[(test_df['Title']== 'Mr') & (test_df['Age'].isnull()), 'Age'] = 32
test_df.loc[(test_df['Title']== 'Mrs') & (test_df['Age'].isnull()), 'Age'] = 39
test_df.loc[(test_df['Title']== 'other') & (test_df['Age'].isnull()), 'Age'] = 44
test_df.loc[(test_df['Title']== 'Master') & (test_df['Age'].isnull()), 'Age'] = 7


# In[ ]:


test_df.isnull().sum().sort_values(ascending=False)


# #### Filling in missing values in train set for Embarked 
# Filling in missing values with most_frequent i.e. S

# In[ ]:


df['Embarked'] = df['Embarked'].fillna('S')


# #### Filling in missing values in test set for Fare

# In[ ]:


test_df.loc[test_df['Fare'].isnull()]


# In[ ]:


# Finding out the mean Fare for Pclass=3

test_df[test_df['Pclass']==3]['Fare'].mean()


# In[ ]:


test_df['Fare'] = test_df['Fare'].fillna(12.46)


# #### Cabin

# In[ ]:


print('Percentage of cabin values missing in train set:', 100.0*df['Cabin'].isnull().sum() / len(df))
print('Percentage of cabin values missing in test set:', 100.0*test_df['Cabin'].isnull().sum() / len(df))


# Too many values for the Cabin feature are missing. Also, it isn't a very useful feature. So drop this feature.

# In[ ]:


print('Missing values for train set')
print(df.isnull().sum().sort_values(ascending=False))
print('----------------')
print('Missing values for test set')
print(test_df.isnull().sum().sort_values(ascending=False))


# All missing values have been treated apart from Cabin which will be dropped later

# ### Feature Engineering continued...

# In[ ]:


df.head()


# #### Grouping ages in Age feature and assigning values based on their survival rate

# In[ ]:


# Creating a new column for age groups

df['AgeGroup'] = pd.cut(df['Age'],5)


# In[ ]:


df[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean().sort_values('Survived', ascending=False)


# Assigning values based on the above age groups. The groups with higher survival rate will be assigned a higher number. So <=16 will be 4, >32 but <=48 will be 3 and so on and so forth

# In[ ]:


df.loc[df['Age'] <= 16, 'Age'] = 4
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 3
df.loc[(df['Age'] >48) & (df['Age'] <= 64), 'Age'] = 2
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 64), 'Age'] = 0


# In[ ]:


df = df.drop('AgeGroup', axis=1)


# for the test set, use the age group results from the train set itself as the test set does not have the target class to be able to do the same analysis

# In[ ]:


test_df.loc[test_df['Age'] <= 16, 'Age'] = 4
test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 3
test_df.loc[(test_df['Age'] >48) & (test_df['Age'] <= 64), 'Age'] = 2
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1
test_df.loc[(test_df['Age'] > 64), 'Age'] = 0


# In[ ]:


df.head()


# #### Grouping fares in Fare feature and assigning values based on their survival rate

# In[ ]:


df[['Fare','Pclass']].groupby('Pclass', as_index=False).mean()


# In[ ]:


df['Fare'].min()


# In[ ]:


df['Fare'].max()


# In[ ]:


df['fareband'] = pd.cut(df['Fare'], 4)


# In[ ]:


df[['fareband', 'Survived']].groupby('fareband', as_index=False).mean().sort_values('Survived', ascending=False)


# In[ ]:


df.loc[(df['Fare'] >= 384), 'Fare'] = 3
df.loc[(df['Fare'] >= 256) & (df['Fare'] < 384), 'Fare'] = 2
df.loc[(df['Fare'] >=128) & (df['Fare'] < 256), 'Fare'] = 1
df.loc[df['Fare'] < 128, 'Fare'] = 0


# In[ ]:


df = df.drop('fareband', axis=1)


# In[ ]:


# Repeating the steps for the test set

test_df.loc[(test_df['Fare'] >= 384), 'Fare'] = 3
test_df.loc[(test_df['Fare'] >= 256) & (test_df['Fare'] < 384), 'Fare'] = 2
test_df.loc[(test_df['Fare'] >=128) & (test_df['Fare'] < 256), 'Fare'] = 1
test_df.loc[test_df['Fare'] < 128, 'Fare'] = 0


# In[ ]:


df.head()


# #### Combining SibSp and Parch to create FamilySize feature

# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1


# The logic adding adding 1 is for solo passengers that don't have any siblings/spouses or parents/children. This feature takes care of all the information in the features SibSp and Parch. We can effectively drop those two features now.

# In[ ]:


# lets take a final look at our dataframe before processing it further

df.head()


# Looks good. 

# In[ ]:


df.info()


# ### Dropping redundant features

# In[ ]:


drop_cols = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']


# In[ ]:


df = df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)


# In[ ]:


df.head()


# ### Creating dummies for all features

# In[ ]:


dummy_cols = ['Pclass','Sex', 'Age',  'Fare', 'Embarked', 'Title', 'FamilySize']
prefix_cats = ['pcl', 'sex', 'age', 'fare', 'emb', 'title', 'fsize']

df = pd.get_dummies(df, columns=dummy_cols, prefix=prefix_cats, drop_first=True)
test_df = pd.get_dummies(test_df, columns=dummy_cols, prefix=prefix_cats, drop_first=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


test_df.shape


# ### Separating features and target

# In[ ]:


X = df.drop('Survived', axis=1)
y = df['Survived']


# # Predictive Modeling

# The basic approach for predictive modeling is as follows:
# - Initialize all the predictors and fit the training data
# - Use cross_val_predict instead of splitting the data as it's quite a small dataset
# - Generate accuracy score for every model
# - Use GridSearachCV for hyperparameter tuning
# - Create parameter grids for grid search
# - Run grid search and find the best estimator
# - Fit the training data this time on the best estimator
# - Use grid search on the best estimator and generate accuracy score
# - Append the model name, accuracy score, and accuracy score on best estimator to a dataframe for comparison
# - Select the model with the best score

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[ ]:


# Creating an empty dataframe to add model predictions for comparison

pred_df = pd.DataFrame()


# In[ ]:


# initialize all the predictors and fit the training data

log_clf = LogisticRegression(random_state=42)
log_clf.fit(X, y)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X, y)

svc_clf = SVC(random_state=42)
svc_clf.fit(X, y)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X, y)

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X, y)

extra_clf = ExtraTreesClassifier(random_state=42)
extra_clf.fit(X, y)

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X, y)


# In[ ]:


# cross_val_predict and generate accuracy scores for all the predictors

log_preds = cross_val_predict(log_clf, X, y, cv=10)
log_acc = accuracy_score(y, log_preds)

sgd_preds = cross_val_predict(sgd_clf, X, y, cv=10)
sgd_acc = accuracy_score(y, sgd_preds)

svc_preds = cross_val_predict(svc_clf, X, y, cv=10)
svc_acc = accuracy_score(y, svc_preds)

tree_preds = cross_val_predict(tree_clf, X, y, cv=10)
tree_acc = accuracy_score(y, tree_preds)

forest_preds = cross_val_predict(forest_clf, X, y, cv=10)
forest_acc = accuracy_score(y, forest_preds)

extra_preds = cross_val_predict(extra_clf, X, y, cv=10)
extra_acc = accuracy_score(y, extra_preds)

gb_preds = cross_val_predict(gb_clf, X, y, cv=10)
gb_acc = accuracy_score(y, gb_preds)


# In[ ]:


print('log_clf', log_acc)
print('sgd_clf', sgd_acc)
print('svc_clf', svc_acc)
print('tree_clf', tree_acc)
print('forest_clf', forest_acc)
print('extra_clf', extra_acc)
print('gb_clf', gb_acc)


# #### Hyperparameter tuning

# In[ ]:


# Generating paramater grids for predictors

log_param = [
    {#'penalty':['l1', 'l2', 'elasticnet'],
    'C':[0.001, 0.01, 0.1, 1.0, 10.0]
    }
]

sgd_param = [
    {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    #'penalty':['l1', 'l2', 'elasticnet']
    }
]

svc_param = [
    {'C':[0.001, 0.01, 0.1, 1.0, 10.0],
    'gamma':[0.001, 0.01, 0.1, 1.0],
    'kernel':['rbf', 'sigmoid']}
]

tree_param = [
    {'max_depth':[2,4,8,12,16,20,30],
    'min_samples_split':[2,4,6,8,10],
    'min_samples_leaf':[2,4,6,8,10]
    }
]

forest_param = [
    {'max_depth':[2,4,8,12,16,20],
    'min_samples_split':[2,4,6,8,10],
    'min_samples_leaf':[2,4,6,8,10],
    'n_estimators':[100,200,300]}
]

extra_param = [
    {'max_depth':[2,4,8,12,16,20,30],
    'min_samples_split':[2,4,6,8,10],
    'min_samples_leaf':[2,4,6,8,10]}
]

gb_param = [
    {'max_depth':[2,8,16,20],
    'min_samples_split':[2,4,6,10],
    'min_samples_leaf':[2,4,6,10],
    'learning_rate':[0.01, 0.05, 0.1],
    'n_estimators':[100,200,300],
    'subsample':[0.5, 0.8, 1.0]}
]


# In[ ]:


log_grid = GridSearchCV(log_clf, log_param, cv=5)
log_grid.fit(X, y)


# In[ ]:


log_best = log_grid.best_estimator_


# In[ ]:


sgd_grid = GridSearchCV(sgd_clf, sgd_param, cv=5)
sgd_grid.fit(X, y)


# In[ ]:


sgd_best = sgd_grid.best_estimator_


# In[ ]:


svc_grid = GridSearchCV(svc_clf, svc_param, cv=5)
svc_grid.fit(X, y)


# In[ ]:


svc_best = svc_grid.best_estimator_


# In[ ]:


tree_grid = GridSearchCV(tree_clf, tree_param, cv=5)
tree_grid.fit(X, y)


# In[ ]:


tree_best = tree_grid.best_estimator_


# In[ ]:


forest_grid = GridSearchCV(forest_clf, forest_param, cv=5, verbose=1, n_jobs=-1)
forest_grid.fit(X, y)


# In[ ]:


forest_best = forest_grid.best_estimator_


# In[ ]:


extra_grid = GridSearchCV(extra_clf, extra_param, cv=5, verbose=1, n_jobs=-1)
extra_grid.fit(X, y)


# In[ ]:


extra_best = extra_grid.best_estimator_


# In[ ]:


gb_grid = GridSearchCV(gb_clf, gb_param, cv=5, verbose=1, n_jobs=-1)
gb_grid.fit(X, y)


# In[ ]:


gb_best = gb_grid.best_estimator_


# In[ ]:


log_best.fit(X, y)


# In[ ]:


sgd_best.fit(X, y)


# In[ ]:


svc_best.fit(X, y)


# In[ ]:


tree_best.fit(X, y)


# In[ ]:


forest_best.fit(X, y)


# In[ ]:


extra_best.fit(X, y)


# In[ ]:


gb_best.fit(X, y)


# In[ ]:


log_best_preds = cross_val_predict(log_best, X, y, cv=10)
log_best_acc = accuracy_score(y, log_best_preds)

sgd_best_preds = cross_val_predict(sgd_best, X, y, cv=10)
sgd_best_acc = accuracy_score(y, sgd_best_preds)

svc_best_preds = cross_val_predict(svc_best, X, y, cv=10)
svc_best_acc = accuracy_score(y, svc_best_preds)

tree_best_preds = cross_val_predict(tree_best, X, y, cv=10)
tree_best_acc = accuracy_score(y, tree_best_preds)

forest_best_preds = cross_val_predict(forest_best, X, y, cv=10)
forest_best_acc = accuracy_score(y, forest_best_preds)

extra_best_preds = cross_val_predict(extra_best, X, y, cv=10)
extra_best_acc = accuracy_score(y, extra_best_preds)

gb_best_preds = cross_val_predict(gb_best, X, y, cv=10)
gb_best_acc = accuracy_score(y, gb_best_preds)


# In[ ]:


pred_df = pred_df.append({'b.Best Estimtor Accuracy': log_best_acc, 'b.Accuracy': log_acc, 'a.Model':'log_clf'}, ignore_index=True)
pred_df = pred_df.append({'b.Best Estimtor Accuracy': sgd_best_acc, 'b.Accuracy': sgd_acc, 'a.Model':'sgd_clf'}, ignore_index=True)
pred_df = pred_df.append({'b.Best Estimtor Accuracy': svc_best_acc, 'b.Accuracy': svc_acc, 'a.Model':'svc_clf'}, ignore_index=True)
pred_df = pred_df.append({'b.Best Estimtor Accuracy': tree_best_acc, 'b.Accuracy': tree_acc, 'a.Model':'tree_clf'}, ignore_index=True)
pred_df = pred_df.append({'b.Best Estimtor Accuracy': forest_best_acc, 'b.Accuracy': forest_acc, 'a.Model':'forest_clf'}, ignore_index=True)
pred_df = pred_df.append({'b.Best Estimtor Accuracy': extra_best_acc, 'b.Accuracy': extra_acc, 'a.Model':'extra_clf'}, ignore_index=True)
pred_df = pred_df.append({'b.Best Estimtor Accuracy': gb_best_acc, 'b.Accuracy': gb_acc, 'a.Model':'gb_clf'}, ignore_index=True)


# #### Comparing accuracy scores

# In[ ]:


pred_df


# Looks like svc_clf and gb_clf are performing best with sgd_clf, forest_clf, and extra_clf close behind it.
# Outputting predictions from svc_clf and gb_clf to a csv document for submission. When the scores for multiple models are similar, due to overfitting either model may perform better on the unseen test dataset. So experimenting with both.
# 
# *I'm also going to submit predictions from the Random Forest model as in my experience random forests tend perform very well on unseen test datasets

# In[ ]:


svc_test_preds = svc_best.predict(test_df)


# In[ ]:


gb_test_preds = gb_best.predict(test_df)


# In[ ]:


forest_test_preds = forest_best.predict(test_df)


# In[ ]:


submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')
submission['Survived'] = svc_test_preds
submission.to_csv('svc_final_submission.csv')


# In[ ]:


submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')
submission['Survived'] = gb_test_preds
submission.to_csv('gb_final_submission.csv')


# In[ ]:


submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')
submission['Survived'] = forest_test_preds
submission.to_csv('forest_final_submission.csv')

