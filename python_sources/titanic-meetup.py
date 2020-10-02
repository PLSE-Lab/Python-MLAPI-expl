#!/usr/bin/env python
# coding: utf-8

# # Goal:
# ### Use a machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

# In[ ]:


print('Hello World')


# # Workflow:
# 1. Load data.
# 2. Exploratory data analysis.
# 3. Clean data.
#     1. Check if data types are correct.
#     2. Feature engineering.
#     3. One-hot encoding of categorical features.
# 4. Train/test and cross validation.
# 5. Training and predicting ML models.
# 6. Submitting predictions.
# 7. Further improvements.
# 8. Hyperparameter tuning.
# 

# In[ ]:


# data manipulation
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization
import seaborn as sns

# handling missing values
import missingno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# machine learning regression
from sklearn.ensemble import RandomForestRegressor

# machine learning classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# evaluating ML models
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV


# In[ ]:


# setup figure sizes
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]


# # Load data

# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=[0])
df_test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col=[0])


# # EDA

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.plot(kind='hist', y='Survived')


# ## Questions
# - Age
# - Sex
# - Cabin
# - Pclass

# In[ ]:


df_train.loc[df_train['Survived'] == 1].plot(kind='kde', y='Age')


# In[ ]:


df_train.loc[df_train['Survived'] == 0].plot(kind='kde', y='Age')


# In[ ]:


help(sns.barplot)


# In[ ]:


sns.distplot(df_train['Se', y='Sex')


# In[ ]:


sns.barplot(data=df_train, y='Sex', hue='Survival')


# ## What does each column mean?

# In[ ]:


df_train.columns


# **What might influence survival?**
# - Having parents or children
# - Which level of the boat you are in
# - How close you are to the impact with iceberg
# - How close you are to flotation devices/boats

# In[ ]:


df_train.info()


# ### Observations
# **Missing values**
# Age and Cabin have many missing values. Embarked has a couple of missing values... why?
# 
# **Which variables should be categorical?**
# - Survived is a yes or no. 
# - Pclass contains three classes. 
# - Name is text... can info be extracted from it?
# - Sex has two. 
# - SibSp has six... what is SibSp? 
# - Parch has seven values, what is it?
# - Ticket is text... can info be extracted from it?
# - Cabin is text... can info be extracted from it?
# - Embarked has three values
# So, categorical variables are: Pclass, Sex, SibSp, Parch, and Embarked.
# 
# **Information in names**
# - Master means boy/baby boy
# 
# 

# # Clean data

# ## Check data types

# In[ ]:


df_train.info()


# In[ ]:


# determine which columns should be categorical
cols_cat = []
for col in df_train: 
    num_unique_vals = df_train[col].unique().size
    if num_unique_vals < 15:
        print(col, df_train[col].dtype, num_unique_vals, df_train[col].unique()) 
        cols_cat.append(col)
    else:
        print(col, df_train[col].dtype, num_unique_vals)
        


# In[ ]:


cols_cat


# In[ ]:


cols_cat.remove('Survived')


# In[ ]:


cols_cat


# In[ ]:


# convert features on train and test set to category
df_train[cols_cat] = df_train[cols_cat].astype('category')
df_test[cols_cat] = df_test[cols_cat].astype('category')


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train.select_dtypes('object')


# In[ ]:


df_train.loc[df_train.Cabin.isnull()]


# <div class="alert alert-info">
#     <h1 class="text-left">
#         What can we do with the remaining "object" features?
#     </h1>
#     <p class="lead">
#         These text data are not in a format from which an algorithm can learn.
#     </p>
# </div>

# ## Feature engineering
# - Married?
# - Has family?
# - Big family?
# - No family?
# - Maiden name?
# - Number of Mrs.
# - Is dependent?
# - How to find family: same cabin, same last name, SibSp and Parch add up, same Ticket...

# In[ ]:


df_train.select_dtypes('object')


# ### Name

# ### Ticket

# ### Cabin

# In[ ]:


df_train.Cabin


# ## _What do these letters mean?_

# ![decks](https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Titanic_cutaway_diagram.png/440px-Titanic_cutaway_diagram.png)

# In[ ]:


df_train.Cabin.str.replace('[^a-zA-Z]', '').value_counts()


# In[ ]:


df_train.Cabin.str.replace('[^a-zA-Z]', '').apply(lambda x: ', '.join(set(x)) if not pd.isnull(x) else np.nan).value_counts()
# I can handle this using one-hot encoding... if sample contains F or G or E... set column value to 1 else 0.


# In[ ]:


df_train['cabin_letter'] = df_train.Cabin.str.replace('[^a-zA-Z]', '').apply(lambda x: ', '.join(set(x)) if not pd.isnull(x) else np.nan)
df_train['cabin_letter'] = df_train['cabin_letter'].astype('category')


# In[ ]:


df_train


# In[ ]:


df_test['cabin_letter'] = df_test.Cabin.str.replace('[^a-zA-Z]', '').apply(lambda x: ', '.join(set(x)) if not pd.isnull(x) else np.nan)
df_test['cabin_letter'] = df_test['cabin_letter'].astype('category')


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# <div class="alert alert-info">
#     <h1>Before creating a model, we need to transform categorical values and handle missing values.</h1>
#     <p>What should we do with cabin_letter?</p>
# </div>

# In[ ]:


sns.distplot(df_train['Age'])


# In[ ]:


df_train['Age'].mean()


# In[ ]:


df_train['Age'].median()


# In[ ]:


df_train.info()


# ## One-hot encoding categorical variables
# 
# Useful links
# - [Kaggle's guide](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding)

# In[ ]:


df_train.info()


# In[ ]:


df_train.select_dtypes(['category', np.number]).info()


# In[ ]:


df_train_1hot = pd.get_dummies(df_train.select_dtypes(['category', np.number]))


# In[ ]:


df_test_1hot = pd.get_dummies(df_test.select_dtypes(['category', np.number]))


# In[ ]:


pd.get_dummies(df_train)


# In[ ]:


df_train_1hot.head()


# In[ ]:


df_test_1hot.head()


# In[ ]:


# find missing columns
set(df_train_1hot.columns) ^ set(df_test_1hot.columns)


# In[ ]:


# make sure columns align
df_train_1hot, df_test_1hot = df_train_1hot.align(df_test_1hot,
                                                  join='left', 
                                                  axis=1)


# In[ ]:


# check if columns of train and test are the same
df_train_1hot.columns == df_test_1hot.columns


# In[ ]:


df_train_1hot.head()


# In[ ]:


df_test_1hot.head()


# In[ ]:


df_test_1hot['cabin_letter_T'] = 0


# ## Handling missing data
# - Three different kinds of missing data.
# - Strategies for handling missing data are:
#     - Complete-case analysis.
#     - Fill missing values with mean, median, mode, or another constant value "single imputation".
#     - Stochastic imputation.
#     - Machine learning to predict missing values.
#     - Forward/back fill if time series

# ### Fill in missing values in Age column

# In[ ]:


df_train_1hot['Age']


# In[ ]:


# split data into train and test
idx_missing = df_train_1hot['Age'].isnull()


# In[ ]:


# split data into train and missing
X_train = df_train_1hot.drop('Age', axis=1).loc[~idx_missing]
X_missing = df_train_1hot.drop('Age', axis=1).loc[idx_missing]
y_train = df_train_1hot['Age'].loc[~idx_missing]
print(X_train.shape)
print(X_missing.shape)
print(y_train.shape)


# In[ ]:


X_train.info()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'RandomForestRegressor')


# Random forest --> non-linear

# In[ ]:


# train classifier
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)


# In[ ]:


y_pred = reg.predict(X_missing)


# In[ ]:


y_pred


# In[ ]:


y_pred = y_pred.round()


# In[ ]:


# compare ages to predictions
sns.distplot(df_train_1hot['Age'].loc[~idx_missing])


# In[ ]:


sns.distplot(y_pred)


# In[ ]:


df_train_1hot['Age'].loc[idx_missing] = y_pred


# In[ ]:


df_train_1hot


# In[ ]:


df_test_1hot.loc[df_test_1hot['Fare'].isnull()]


# In[ ]:


df_test_1hot.loc[df_test_1hot['Fare'].isnull()] = df_test_1hot['Fare'].mean()


# In[ ]:


# fill in age in df_test
idx_missing = df_test_1hot['Age'].isnull()
X_train = df_test_1hot.drop(['Survived', 'Age'], axis=1).loc[~idx_missing]
X_missing = df_test_1hot.drop(['Survived', 'Age'], axis=1).loc[idx_missing]
y_train = df_test_1hot['Age'].loc[~idx_missing]
reg = RandomForestRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_missing).round()
df_test_1hot['Age'].loc[idx_missing] = y_pred


# In[ ]:


df_train_1hot.info()


# In[ ]:


df_test_1hot.info()


# In[ ]:





# # Predicting survival with a baseline model
# Sklearn has many classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# In[ ]:


X = df_train_1hot.drop('Survived', axis=1).values
X_test = df_test_1hot.drop('Survived', axis=1).values
y = df_train_1hot['Survived'].values


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[ ]:


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)


# In[ ]:


accuracy_score(y_val, y_pred)


# In[ ]:


print(classification_report(y_val, y_pred))


# In[ ]:


clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)


# In[ ]:


accuracy_score(y_val, y_pred)


# In[ ]:


print(classification_report(y_val, y_pred))


# ## Cross validation
# ![cross validation](https://ethen8181.github.io/machine-learning/model_selection/img/kfolds.png)

# In[ ]:


# Create a StratifiedKFold object
str_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Loop through each split
fold = 0
for idx_train, idx_val in str_kf.split(X, y):
    # Obtain training and testing folds
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(X_train.shape))
    print('Number of survivors: {}\n'.format(sum(y_train == 1)))
    fold += 1


# In[ ]:


df_train['Survived'].value_counts()


# ### Random forest classifier

# In[ ]:


# Loop through each split
fold = 0
list_accuracy = []
for idx_train, idx_val in str_kf.split(X, y):
    # Obtain training and testing folds
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(X_train.shape))
    print('Number of survivors: {}\n'.format(sum(y_train == 1)))
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    list_accuracy.append(accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    fold += 1


# In[ ]:


list_accuracy


# In[ ]:


cross_val_score(clf, X, y, cv=str_kf, scoring='accuracy')


# ### Other classifiers...

# ### GaussianProcessClassifier

# In[ ]:


from sklearn.gaussian_process import GaussianProcessClassifier
clf = GaussianProcessClassifier()
cross_val_score(clf, X, y, cv=str_kf, scoring='accuracy')


# ### Nearest neighbors
# - "Curse of dimensionality"

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
cross_val_score(clf, X, y, cv=str_kf, scoring='accuracy')


# ### SVC

# In[ ]:


from sklearn.svm import SVC
clf = SVC(kernel='linear')
cross_val_score(clf, X, y, cv=str_kf, scoring='accuracy')


# ### XGBoost

# In[ ]:


import xgboost as xgb
params = {'objective': 'reg:linear',
          'max_depth': 5,
          'silent': 1}
# Loop through each split
fold = 0
list_accuracy = []
for idx_train, idx_val in str_kf.split(X, y):
    # Obtain training and testing folds
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val)
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(X_train.shape))
    print('Number of survivors: {}\n'.format(sum(y_train == 1)))
    clf = xgb.train(params=params, dtrain=dtrain)
    y_pred = clf.predict(dval).round()
    list_accuracy.append(accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    fold += 1


# In[ ]:


list_accuracy


# # Submitting predictions

# In[ ]:


# predict on test data
clf = SVC(kernel='linear')
clf.fit(X, y)
y_pred = clf.predict(X_test)


# In[ ]:


df_pred = pd.DataFrame(y_pred, index=df_test.index, columns=['Survived'])


# In[ ]:


df_pred


# In[ ]:


df_pred.to_csv('predictions_SVC.csv')


# # Improving our score
# - More feature engineering.
#     - Extracting information from Name and Ticket variables.
#     - Use a "family" category.
# - Different ways to fill in missing values.
#     - Flag missing cabins as "Missing" value.
#     - Target encoding.
# - Hyperparameter tuning.
# - Ensemble models.

# # Hyperparameter tuning

# In[ ]:


help(LogisticRegression)


# In[ ]:


help(GridSearchCV)


# In[ ]:


param_grid = {
    'solver': ['newton-cg', 'liblinear'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.5, 1, 10],
    'random_state': [42],
    'max_iter': [500, 1000, 4000]
}
grid_search = GridSearchCV(LogisticRegression(), param_grid)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


cross_val_score(LogisticRegression(max_iter=300), X, y, cv=str_kf, scoring='accuracy')


# In[ ]:


clf = LogisticRegression(**grid_search.best_params_)
cross_val_score(clf, X, y, cv=str_kf, scoring='accuracy')


# In[ ]:


clf.fit(X, y)
y_pred = clf.predict(X_test)


# In[ ]:


df_pred = pd.DataFrame(y_pred, index=df_test.index, columns=['Survived'])


# In[ ]:


df_pred


# In[ ]:


df_pred.to_csv('predictions_hyperparameter_tuning_grid_search.csv')


# In[ ]:


df_train


# In[ ]:


sns.countplot(data=df_train, x='Sex', hue='Survived')


# In[ ]:


sns.countplot(data=df_train, x='Pclass', hue='Survived')


# # Some notes

# # Categorizing and handling missing data on train and test sets
# 
# The purpose when addressing missing data is to correctly reproduce the variance/covariance matrix we would have observed had our data not had any missing information.
# 
# There are three categories of missing data:
# 1. Missing completely at random (MCAR): the reason for missingness has nothing to do with other variables.
#     - We can impute these values with test statistics.
#     - We can predict these values.
# 2. Missing at random (MAR): the reason for missingness has to do with one or more variables
# 3. Missing not at random (MNAR)
# ![handling missing data](https://miro.medium.com/max/1400/1*_RA3mCS30Pr0vUxbp25Yxw.png)
# Useful links:
# - [Sklearn docs](https://scikit-learn.org/stable/modules/impute.html)
# - [Blog post](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)
# - [Book: Applied missing data analysis](http://hsta559s12.pbworks.com/w/file/fetch/52112520/enders.applied)

# In[ ]:





# # Other possible feature engineering steps

# 

# ## Go back and redo Cabin values
# We never imputed values for cabin. Instead, we one-hot encoded our data, which set missing cabin_letter values to 0. Could we improve our predictions if we go back and imputed cabins after we replaced the missing values in the Age column?

# In[ ]:


cols_cabin = [i for i in df_train_1hot.columns if i.startswith('cabin_letter')]


# In[ ]:


df_train_1hot = df_train_1hot.loc[df_train.cabin_letter.isnull()].drop(cols_cabin, axis=1)
df_test_1hot = df_test_1hot.loc[df_test.cabin_letter.isnull()].drop(cols_cabin, axis=1)


# In[ ]:


# add cabin_letter back into data
df_train_1hot['cabin_letter'] = df_train['cabin_letter']


# In[ ]:


# predict cabin letter
X_train = df_train_1hot.loc[df_train['cabin_letter']]
clf = RandomForestClassifier()
clf.fit


# In[ ]:


df_train.shape


# In[ ]:


df_train_1hot.shape


# In[ ]:


df_train['cabin_letter'].value_counts()


# In[ ]:


# split data into train and test
idx_missing = df_train_1hot['cabin_letter'].isnull()


# In[ ]:


# split data into train and missing
X_train = df_train_1hot.drop('cabin_letter', axis=1).loc[~idx_missing]
X_missing = df_train_1hot.drop('cabin_letter', axis=1).loc[idx_missing]
y_train = df_train_1hot['cabin_letter'].loc[~idx_missing]
print(X_train.shape)
print(X_missing.shape)
print(y_train.shape)


# In[ ]:


X_train.info()


# In[ ]:


# train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)


# In[ ]:


y_pred = reg.predict(X_missing)


# In[ ]:


y_pred.round()


# In[ ]:


# compare ages to predictions
sns.distplot(df_train_1hot['Age'].loc[~idx_missing])


# In[ ]:


sns.distplot(y_pred)


# In[ ]:


df_train_1hot['Age'].loc[idx_missing] = y_pred


# In[ ]:


df_train_1hot.info()


# ## Ticket number

# In[ ]:


ticket_num = pd.Series([i[-1] for i in df_train.Ticket.str.split()], index=df_train.index, name='Ticket number')


# In[ ]:


ticket_num


# In[ ]:


df_train.loc[~ticket_num.str.isnumeric()]


# In[ ]:


ticket_num.loc[~ticket_num.str.isnumeric()] 


# In[ ]:


ticket_num.loc[~ticket_num.str.isnumeric()] = np.NaN


# In[ ]:


ticket_num = ticket_num.astype(float)


# In[ ]:


sns.distplot(ticket_num)


# In[ ]:


df_train['ticket_number'] = ticket_num


# In[ ]:


# do the same for df_test
ticket_num_test = pd.Series([i[-1] for i in df_test.Ticket.str.split()], index=df_test.index, name='Ticket number')
ticket_num_test.loc[~ticket_num_test.str.isnumeric()] = np.NaN
ticket_num_test = ticket_num_test.astype(float)
df_test['ticket_number'] = ticket_num_test

