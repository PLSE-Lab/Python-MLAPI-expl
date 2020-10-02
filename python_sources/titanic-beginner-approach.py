#!/usr/bin/env python
# coding: utf-8

# This is a submission for Kaggle Competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview).
# 
# This version give test accuracy score of `0.79425` which resulted rank `3435` out of `22814` competitors in the leaderboard (**top 15%**). 
# Not bad for my first attempt on Kaggle Competition. 
# 
# Any suggestions on the approach are most welcome. Please give upvote if you like this notebook.
# 
# ![leaderboard27may.png](attachment:leaderboard27may.png)
# 
# Screenshot taken on 27 May 2020

# ## Outline
# 1. **Loading Data**
# 2. **Analysing Raw Features** : simple histogram, Information Value
# 3. **Engineering and Selecting Features** : handle missing values and features transformation
# 4. **Selecting model** : test few models and select the highest AUC model
# 5. **Fine-tuning model** : grid search with cross validation
# 6. **Predicting test set**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Function

# Function that will be used in below notebook.

# In[ ]:


# Calculate information value
def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())


    iv = data['IV'].sum()
    # print(iv)

    return iv, data


# In[ ]:


def add_log_transform(df, feat_list):
    # Generate log transform
    for feat in feat_list:
        df[feat+'_log'] = np.log1p(df[feat])
    return df


# In[ ]:


def add_age_bin(df):
    # Generate Age_bin
    df.loc[ df['Age'] <= 12, 'Age_bin'] = 0
    df.loc[(df['Age'] > 12) & (df['Age'] <= 24), 'Age_bin'] = 1
    df.loc[(df['Age'] > 24) & (df['Age'] <= 36), 'Age_bin'] = 2
    df.loc[(df['Age'] > 36) & (df['Age'] <= 48), 'Age_bin'] = 3
    df.loc[(df['Age'] > 48) & (df['Age'] <= 60), 'Age_bin'] = 4
    df.loc[ df['Age'] > 60, 'Age_bin'] = 5
    
    return df


# In[ ]:


def get_title(name):
    # Extract title from name
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def add_title(df):
    # Add new feature title
    df['Title'] = df['Name'].apply(get_title)
    
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    return df


# ## Loading Data

# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[ ]:


train_data.info()


# Age and Embarked have some missing values, we might need to impute them if they are good signals.
# 
# Cabin has too many missing values, we might need to drop them for model training.

# In[ ]:


test_data.info()


# ## Analysing Raw Features

# Simple distribution analysis of raw features and Information Value analysis

# In[ ]:


tgt = 'Survived'


# In[ ]:


train = train_data.copy()


# In[ ]:


train.drop('PassengerId',axis=1).groupby(tgt).hist()


# Noteable pattern:
# * Children (young age people) tend to have higher chance to survive. Hence, Age is potentially a good model feature.
# * Higher Pclass tend to have higher chance to survive. Hence, Pclass is potentially a good model feature.
# * Distribution of Age and Fare look skewed. Apply log transform to those features to make the distribution closer to normal distribution.

# In[ ]:


train.describe(include=['O'])


# Noteable pattern:
# * Name is completely unique, hence Name is potentially not a good feature | UPDATE: After looking at others' notebook, it turn out Title extracted from the Bame turns out a good feature.
# * Ticket is almost completely unique, hence Ticket is potentially not a good feature.
# * Cabin has too much missing values, hence Cabin will be dropped for model training.

# In[ ]:


raw_feat = train.columns.to_list()
raw_feat.pop(1) #remove Survived


# In[ ]:


ivs={}
for feat in raw_feat:
    iv, data = calc_iv(train,feat, tgt)
    ivs[feat] = iv
ivs
    


# Noteable insights:
# * Sex, Fare, Pclass have suspicious high predictive power
# * Age has a strong predictive power.
# * Majority of the rest has medium predictive power.

# ## Engineering and Selecting Features

# Dirty approach of feature engineering to generate many features, and later select which one is highly correlated to the target and weakly correlated with other features to reduce collinearity.

# In[ ]:


train = train_data.copy()

# Group attributes based on data type
num_attribs= ['Age','Fare','Age_median_by_Pclass']
cat_attribs = ['Pclass', 'Sex', 'Embarked']
omit_attribs = ['Cabin', 'Name', 'Ticket', 'PassengerId']

# Impute age with median values within its Pclass
age_imputer = train.groupby('Pclass').Age.median()
train['imputed_age'] = train.Pclass.apply(lambda x: age_imputer[x])
train['Age_median_by_Pclass'] = train.Age
train['Age_median_by_Pclass'].fillna(train['imputed_age'], inplace=True)
train = train.drop('imputed_age',axis=1) #drop the dummy column

# Alternate way to impute age by taking its median value
train.Age.fillna(train.Age.median(), inplace=True)

# Generate Age_bin
train = add_age_bin(train)

# Generate Log Transform of numerical attributes
train = add_log_transform(train, num_attribs)

# Generate title
train = add_title(train)

# Generate IsAlone and FamilySize
train['IsAlone'] = (train.Parch + train.SibSp == 0)*1
train['FamilySize'] = train.Parch + train.SibSp + 1

# Generate HasCabin
train['HasCabin'] = ~train.Cabin.isna()

# Impute Embarked with most frequent values
train.Embarked.fillna(train.Embarked.mode(), inplace=True)

# Onehot encode categorical variables
train_transformed = pd.get_dummies(train[cat_attribs]).join(train.drop(cat_attribs + omit_attribs,axis=1))


# In[ ]:


train_transformed[['Fare_log','Age_log','Age_median_by_Pclass_log']].hist()


# Log transform of Age looks closer to normal distribution, whereas log transform of Fare is still skewed.

# In[ ]:


# Feature correlation with target
train_transformed.corr()['Survived'].sort_values(ascending=False)


# In[ ]:


# Remove duplicate features or features which are less correlated with target
train_transformed.drop(['Sex_male', 'Age','Age_median_by_Pclass','Age_median_by_Pclass_log','Fare'],axis=1, inplace=True)


# In[ ]:


# Feature correlation with target
train_transformed.corr()['Survived'].sort_values(ascending=False)


# In[ ]:


# Draw correlation heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_transformed.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# There are few features that are highly correlated to each other because they are derived from the same raw features. We decided to keep all features since they have different orders of feature importances / coefficients depends on the model that we select.
# 
# The highly correlated features are namely:
# * Age_bin and Age_log
# * FamilySize & IsAlone and Parch & SibSp
# * Embarked
# 

# ## Selecting model

# Train various algorithms and pick model with highest AUC

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_transformed.drop('Survived',axis=1), train_transformed.Survived, test_size=0.30, random_state=42)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print('AUC', roc_auc_score(y_test,rfc_pred))
print('accuracy: ',accuracy_score(y_test,rfc_pred))

rfc_feat_imp = pd.DataFrame(rfc.feature_importances_, index=X_train.columns.to_list(), columns=['feat_imp'])
rfc_feat_imp.sort_values('feat_imp', ascending=False)


# In[ ]:


# SVC
from sklearn.svm import SVC

svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print('AUC', roc_auc_score(y_test,svc_pred))
print('accuracy: ',accuracy_score(y_test,svc_pred))


# In[ ]:


# # Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('AUC', roc_auc_score(y_test,lr_pred))
print('accuracy: ',accuracy_score(y_test,lr_pred ))

lr_coef = pd.DataFrame(lr.coef_[0], index=X_train.columns.to_list(), columns=['coef'])
lr_coef.sort_values('coef', ascending=False)


# In[ ]:


# # KNeighborClassifier
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
knc_pred = knc.predict(X_test)
print('AUC', roc_auc_score(y_test,knc_pred))
print('accuracy: ',accuracy_score(y_test,knc_pred ))


# In[ ]:


# LightGBM
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
print('AUC', roc_auc_score(y_test,lgbm_pred))
print('accuracy: ',accuracy_score(y_test,lgbm_pred))


# We decided to proceed with SVC due its highest AUC during model selection

# ## Fine-tuning Model

# To fine-tune the model hyperparameter, we do grid search with cross validation

# In[ ]:


param_grid = [
     {'kernel':('linear', 'poly','rbf'), 'C':[1, 10], 'degree':[3,4]}
  ]

svc = SVC(random_state=42)
grid_search = GridSearchCV(svc, param_grid, cv=5,
                           scoring='accuracy',
                           return_train_score=True)
grid_search.fit(train_transformed.drop('Survived',axis=1), train_transformed.Survived)


# In[ ]:


estimator = grid_search.best_estimator_


# ## Predicting Test Data

# Engineer Test Data Features and do prediction

# In[ ]:


test = test_data.copy()

# Impute missing values and onehot encode categorical variable
num_attribs= ['Age','Fare']
cat_attribs = ['Pclass', 'Sex', 'Embarked']
omit_attribs = ['Cabin', 'Name', 'Ticket', 'PassengerId']

# Impute age by taking its median value
test.Age.fillna(test.Age.median(), inplace=True)

# Generate Age_bin
test = add_age_bin(test)

# Impute Fare with median value of its Pclass
fare_imputer = train.groupby('Pclass').Fare.median()
test['imputed_fare'] = test.Pclass.apply(lambda x: fare_imputer[x])
test['Fare'].fillna(test['imputed_fare'], inplace=True)
test = test.drop('imputed_fare',axis=1) #drop the dummy column

# Generate log transform of numerical attributes
test = add_log_transform(test, num_attribs)

# Generate title
test = add_title(test)

# Generate new features IsAlone and FamilySize
test['IsAlone'] = (test.Parch + test.SibSp == 0)*1
test['FamilySize'] = test.Parch + test.SibSp + 1

# Generate new features HasCabin
test['HasCabin'] = ~test.Cabin.isna()

# Impute Embarked with most frequent values
test.Embarked.fillna(train.Embarked.mode(), inplace=True)

# Onehot encode categorical variables
test_transformed = pd.get_dummies(test[cat_attribs]).join(test.drop(cat_attribs + omit_attribs,axis=1))

# Remove duplicate or less correlated features
test_transformed.drop(['Sex_male', 'Age','Fare'],axis=1, inplace=True)


# In[ ]:


test_pred = estimator.predict(test_transformed)


# In[ ]:


test_data['Survived'] = test_pred


# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('prediction.csv', index=False)


# End of notebook
