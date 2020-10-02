#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import feature_selection
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[ ]:


files = ['/kaggle/input/titanic/train.csv', '/kaggle/input/titanic/test.csv']
train_df, test_df = [pd.read_csv(i) for i in files]
df_list = [train_df, test_df]
train_df.head()


# ### Description Overview

# In[ ]:


train_df.info()
print('-' * 40)
test_df.info()


# In[ ]:


train_df[['Survived', 'Age', 'Fare']].describe()


# In[ ]:


train_df.describe(include = ['O'])


# #### Summary
# - There are a few missing values in column Age, Cabin and Fare
# - Survivided rate is only 38.38% in train_df
# - No Name duplicate, Ticket and Cabin have low freq 

# ### Categorical Variables Overview

# In[ ]:


def p_survive(check_list, dataframe):
    '''quickly group the variable to check the survived rate for each categorical variable in the check_list'''
    
    for name in check_list:
        print('{} Survived Possibility: '.format(name))
        print(dataframe[[name, 'Survived']].groupby(
            name, as_index = True).mean().sort_values(
            by = 'Survived', ascending = False).reset_index())
        print('-' * 10)

check_list = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']
p_survive(check_list, train_df)


# ### Summary
# - Sex is highly correlated with survived rate while Embarked is lowerly correlated with survived rate
# - Other variables are correlated with survived rate (next will visualize the data for further information)

# ### Visualization Overview

# In[ ]:


# Sex & Age
g = sns.FacetGrid(train_df, hue = 'Survived', col = 'Sex', height = 3, aspect = 2)
g.map(plt.hist, 'Age', alpha = .5, bins = 20)
g.add_legend()
plt.show()


# In[ ]:


# Sex & Fare
g = sns.FacetGrid(train_df, hue = 'Survived', col = 'Sex', height = 4, aspect = 1.5)
g.map(plt.hist, 'Fare', alpha = .5)
g.add_legend()
plt.show()


# In[ ]:


# Sex & Pclass & Embarked
FacetGrid = sns.FacetGrid(train_df, row='Embarked', height = 3 , aspect = 2)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'Blues')
FacetGrid.add_legend()
plt.show()


# In[ ]:


# Age & Pclass 
g = sns.FacetGrid(train_df, hue = 'Survived', row = 'Pclass', height = 3, aspect = 2.5)
g.map(plt.hist, 'Age', alpha = .5, bins = 20)
g.add_legend()
plt.show()


# ### Data Manipulation

# In[ ]:


df_list = [df.drop(['Ticket', 'Cabin'], axis = 1) for df in df_list]

def manipulate_df(df):
    '''part 1: filling all missing values
       part 2: creating features
       part 3: setting segments for continues variables'''
    
    df.Age.fillna(df.Age.median(), inplace = True)
    df.Fare.fillna(df.Fare.median(), inplace = True)
    df.Embarked.fillna(df.Embarked.mode()[0], inplace = True)
    
    df['FamilySize'] = df.SibSp + df.Parch + 1
    df['Alone'] = [1 if i == 1 else 0 for i in df.FamilySize]
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand = False)
    df['FarePP'] = (df.Fare/df.FamilySize).astype(int)
    
    df['FareRange'] = pd.qcut(df.Fare, 4, labels = [1, 2, 3, 4])
    df['FarePPRange'] = pd.qcut(df.FarePP, 4, labels = [1, 2, 3, 4])
    df['AgeRange'] = pd.cut(df.Age.astype(int), 5, labels = [1, 2, 3, 4, 5])
    
    return df


# In[ ]:


df_list[0] = manipulate_df(df_list[0])

# checking dataset & null values
display(df_list[0].head(3))
print('Titles in the dataset\n', df_list[0].Title.unique())
print('-' * 30)
print(df_list[0].isnull().sum())


# In[ ]:


# creating title groups 
other_titles = ['Jonkheer', 'Col', 'Capt', 'Countess', 'Major', 'Dr', 'Master', 'Rev']
female_titles1 = ['Miss', 'Mlle', 'Ms']
female_titles2 = ['Mme', 'Lady', 'Mrs', 'Dona']
male_titles = ['Mr', 'Don', 'Sir']

def title_replace(df):
    '''grouping titles for Machine Learning preprocessing'''
    
    df['Title'] = df.Title.replace(other_titles, 'Others')
    df['Title'] = df.Title.replace(female_titles1, 'Miss')
    df['Title'] = df.Title.replace(female_titles2, 'Mrs')
    df['Title'] = df.Title.replace(male_titles, 'Mr')
    
    return df


# In[ ]:


df_list[0] = title_replace(df_list[0])

# checking survived rate by grouping new features
check_list2 = ['Title', 'FamilySize', 'Alone', 'AgeRange', 'FareRange', 'FarePPRange']
p_survive(check_list2, df_list[0])


# In[ ]:


# visualizing suivived rate by grouping FamilySize
sns.factorplot('FamilySize','Survived', data = df_list[0], height = 3.5, aspect = 3.5)
plt.show()


# ### Machine Learning

# In[ ]:


# preprocessing dataset
le = LabelEncoder()
def label_encoder(cat_list, coded_list, dataframe):
    '''step 1: encoding all labels in the cat_list in the dataframe 
       step 2: appending new labels into a new list named coded_list
       step 3: return new dataframe and coded_list'''
    
    for col in cat_list:
        dataframe['{}_le'.format(col)] = le.fit_transform(dataframe['{}'.format(col)])
        coded_list.append('{}_le'.format(col))
    return dataframe, coded_list


# In[ ]:


# combining all the features (original & created)
cat_list = check_list + check_list2

coded_list = []
label_encoder(cat_list, coded_list, df_list[0])
df_list[0][cat_list].head(3)


# In[ ]:


# checking the labels for all the features 
X = df_list[0][coded_list]
y = df_list[0].Survived
print('Categorical Columns:\n', cat_list)
print('-' * 40)
print('Encoded Categorical Columns:\n', X.columns.values)


# In[ ]:


# spliting data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# creating a model list to get a better model with highest score
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('KNearest Neighbors', KNeighborsClassifier(n_neighbors = 4)))
models.append(('SGD Classfier', SGDClassifier()))
models.append(('Gaussian NB', GaussianNB()))
models.append(('Linear SVC', LinearSVC()))
models.append(('Random Forest', RandomForestClassifier(max_depth = 5, n_estimators = 200, oob_score = True)))
models.append(('AdaBoost Classifier', AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5), 
                                              n_estimators = 200, algorithm = "SAMME.R")))
models.append(('Bagging Classifier', BaggingClassifier(DecisionTreeClassifier(max_depth = 5), 
                                            n_estimators = 200, bootstrap = True)))

labels = []
results = []
std = []

for label, model in models:
    kfold = KFold(n_splits = 10, shuffle = True)
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'accuracy')
    labels.append(label)
    results.append(round(cv_results.mean() * 100, 2))
    std.append(round(cv_results.std(), 4))

# creating a sorted dataframe to show the highest avg.score model
df_result = pd.DataFrame({'Model': labels, 'Avg_score': results, 'Avg_std': std})
df_result.sort_values(by = 'Avg_score', ascending = False)


# In[ ]:


# tuning the model by excluding the unimportance features from the highest avg.score model 
rf = RandomForestClassifier(max_depth = 5, n_estimators = 200)
rf.fit(X_train, y_train)

df_vips = pd.DataFrame({'Feature': cat_list, 
                       'Importance': rf.feature_importances_})

plt.figure(figsize = (10, 5))
sns.barplot(x = 'Importance', y = 'Feature', 
            data = df_vips.sort_values(by = 'Importance', ascending = False), palette = 'Blues_d')
plt.title('Importance Score', fontsize = 14)
plt.show()


# In[ ]:


# dropping two features with lowest importance score 
X_train_better, X_test_better = [X.drop(['Alone_le', 'Parch_le'], axis = 1) for X in [X_train, X_test]]

# refitting the model 
rf = RandomForestClassifier(max_depth = 5, n_estimators = 200, oob_score = True)
rf.fit(X_train_better, y_train)
y_pred = rf.predict(X_test_better)

print('Better Random Forest Training Set Score: {:.2f}%'.format(rf.score(X_train_better, y_train) * 100))
print('Better Random Forest Predicting Set Score: {:.2f}%'.format(rf.score(X_test_better, y_pred) * 100))


# In[ ]:


# hyperparameters tuning to get the best model params
hp_params = {'max_depth': range(3, 8, 1),
             'n_estimators': range(150, 400, 50),
             'criterion': ['gini', 'entropy']}

rf_sg = GridSearchCV(estimator = RandomForestClassifier(oob_score = True), param_grid = hp_params, cv = 5)
rf_sg.fit(X_train_better, y_train)
y_pred_best = rf_sg.best_estimator_.predict(X_test_better)

print('Best Params: {}\nTraining Set Score: {:.2f}%\nPredicting Score on: {:.2f}%'.format(
    rf_sg.best_params_, rf_sg.best_score_*100, rf_sg.best_estimator_.score(X_test_better, y_pred_best)*100))


# In[ ]:


df_list[1].head()


# In[ ]:


# manipulating test dataset
df_list[1] = manipulate_df(df_list[1])
df_list[1] = title_replace(df_list[1])

# checking the details of the new dataset
display(df_list[1].head())
print('Titles in the dataset\n', df_list[1].Title.unique())
print('-' * 30)
print(df_list[1].isnull().sum())


# In[ ]:


# preprocessing the new dataset
coded_list2 = []
label_encoder(cat_list, coded_list2, df_list[1])

X2 = df_list[1][coded_list2].drop(['Alone_le', 'Parch_le'], axis = 1)
print('Encoded Categorical Columns:\n', X2.columns.values)


# In[ ]:


# final result
y2_pred = rf_sg.best_estimator_.predict(X2)
test_df['Survived'] = y2_pred
submission = test_df[['PassengerId', 'Survived']]

submission.to_csv("Titanic_submit.csv", index = False)


# In[ ]:




