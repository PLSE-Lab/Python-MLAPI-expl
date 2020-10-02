#!/usr/bin/env python
# coding: utf-8

# In[148]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[149]:


# Import all necessary libraries
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder, PolynomialFeatures
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
# Classification Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# In[150]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_Subm = pd.read_csv('../input/gender_submission.csv')
df_all = pd.concat([df_train,df_test])


# In[151]:


df_all.reset_index(inplace=True)


# In[152]:


df_all.drop(columns=['index'],inplace=True)


# In[153]:


df_all.head()


# In[154]:


def Create_Title(df_in):
    df = df_in.copy()
    for i, item in enumerate(df['Name']):
        start = item.find(',')+2
        end = item.find('.')
        df.loc[i,'Title'] = item[start:end]
    return df


# In[155]:


def Create_Surname(df_in):
    df = df_in.copy()
    for i, item in enumerate(df['Name']):
        end = item.find(',')
        df.loc[i,'Surname'] = item[:end]
    return df


# In[156]:


df_all1 = Create_Title(df_all)
df_all1 = Create_Surname(df_all1)


# In[157]:


df_all1.info()


# In[158]:


def Family_Size(df_in):
    df = df_in.copy()
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df.loc[df['Family_Size'] > 1,'SoloTraveller'] = 0
    df.loc[df['Family_Size'] <= 1,'SoloTraveller'] = 1
    df.SoloTraveller = df.SoloTraveller.astype('int')
    return df


# In[159]:


df_all1 = Family_Size(df_all1)


# In[160]:


df_all1.info()


# In[161]:


def fill_Age(df):
    sex = df.Sex.unique().tolist()
    for i in sex:
        for j in range(1, 4):
            df_age = df[(df['Sex'] == i) & (df['Pclass'] == j)]['Age'].dropna()
            df.loc[(df.Age.isnull() & (df.Sex==i) & (df.Pclass==j)), 'Age'] = df_age.mean()


# In[162]:


def fill_Embarked(df):
    df['Embarked'].fillna(df['Embarked'].dropna().mode()[0], inplace=True)


# In[163]:


df_all1[df_all1.Fare == 0]['Fare'].count()


# In[164]:


def ZerotoNan(df):
    df['Fare'].replace(0,np.nan,inplace=True)


# In[165]:


def fill_ZeroFare(df):
    for j in range(1, 4):
        df_fare = df[(df['Fare'] != 0) & (df['Pclass'] == j)]['Fare'].dropna()
        df.loc[((df['Fare'] == 0) & (df['Pclass'] == j)), 'Fare'] = round(df_fare.mean(),2)


# In[166]:


def fill_NaNFare(df):
    for j in range(1, 4):
        df_fare = df[df['Pclass'] == j]['Fare'].dropna()
        df.loc[((df['Fare'].isnull()) & (df['Pclass'] == j)), 'Fare'] = round(df_fare.dropna(0).mean(),2)


# In[167]:


fill_Age(df_all1)
fill_Embarked(df_all1)
# ZerotoNan(df_all1)
# fill_ZeroFare(df_all1)
fill_NaNFare(df_all1)


# In[168]:


df_all1.info()


# In[169]:


drop_col = ['PassengerId','Name','Ticket','Cabin']
df_all1.drop(columns=drop_col, inplace=True)


# In[170]:


df_all1.head()


# In[171]:


def List_Unique(df):
    for col in df.columns.tolist():
        print('Col: {}\t\tTotal no. of unique values: {} '.format(col,df[col].nunique()))        


# In[172]:


List_Unique(df_all1)


# In[173]:


df_all1.groupby('Title')['Age'].count()


# In[174]:


def Clean_Title(df,MISC_list,Mr_list,Miss_list,Mrs_list):
    for item in Mrs_list:
        df.loc[df['Title'] == item,'Title'] = 'Mrs'
    for item in Mr_list:
        df.loc[df['Title'] == item,'Title'] = 'Mr'
    for item in Miss_list:
        df.loc[df['Title'] == item,'Title'] = 'Miss'
    for item in MISC_list:
        df.loc[df['Title'] == item,'Title'] = 'MISC'


# In[175]:


Mr_lt = ['Sir','Major','Col','Capt','Don']
Mrs_lt = ['Mme','the Countess','Dona']
Miss_lt = ['Ms','Lady','Mlle']
MISC_list = ['Rev','Dr','Jonkheer']

Clean_Title(df_all1,MISC_list,Mr_lt,Miss_lt,Mrs_lt)


# In[176]:


df_all1.head()


# In[177]:


df_all1.groupby('Surname')['Pclass'].count()


# In[178]:


df_all1.groupby('Family_Size')['Age'].count()


# In[179]:


train_df = df_all1[:891]
test_df  = df_all1[891:]


# In[180]:


test_df.drop(columns='Survived', inplace=True)


# In[181]:


test_df.head()


# In[182]:


df_test.head()


# In[183]:


def oneGender_Survived(df_in,gender):
    df=df_in.copy()
    df_Survived = pd.DataFrame(index=df.index)
    for item in gender:
        df_Survived.loc[df['Sex'] == item,'Pred_'+item] = 1
        df_Survived['Pred_'+item].fillna(0,inplace=True)
    return df_Survived.astype(int)


# In[184]:


def all_same(df_in):
    df_Survived = pd.DataFrame(index=df_in.index)        
    df_Survived.loc[df_in.index.tolist(),'Died'] = 0
    df_Survived.loc[df_in.index.tolist(),'Alive'] = 1
    return df_Survived.astype(int)


# In[185]:


# sample submission to check which class is favoured in the result
sub_df = pd.DataFrame()
sub_df = pd.concat([sub_df,oneGender_Survived(test_df,['male','female']),all_same(test_df)],axis=1)


# In[186]:


sub_df.head()


# In[187]:


# fare_range = pd.cut(df_all1.Fare, bins=6,labels=[0,1,2,3,4,5],retbins=True)


# In[188]:


# len(fare_range[0])


# In[189]:


# df_all1.tail()


# In[190]:


# df_all1['Fare_bin'] = fare_range[0]
# df_all1['Fare_bin'] = df_all1['Fare_bin'].astype('int64')
# df_all1.head()


# In[191]:


# Age_range = pd.cut(df_all1.Age, bins=3,labels=[0,1,2],retbins=True)


# In[193]:


df_all1['Age_bin'] = 'Adult'
df_all1.loc[df_all1.Age < 18,'Age_bin'] = 'Child'
df_all1.loc[df_all1.Age > 50,'Age_bin'] = 'Old'
df_all1.head()


# In[194]:


List_Unique(df_all1)


# In[195]:


df_all1.drop('Surname', axis=1,inplace=True)


# In[196]:


dummy_cols = ['Sex', 'Embarked', 'Pclass', 'SoloTraveller', 'Age_bin', 'Title']
df_all1 = pd.get_dummies(df_all1, columns=dummy_cols)


# In[197]:


df_all1.head()


# In[198]:


trn_df = df_all1[:891]
tst_df = df_all1[891:]
trn_df.drop(columns='Age', inplace=True)
tst_df.drop(columns=['Survived','Age'], inplace=True)


# In[199]:


trn_df_X = trn_df.drop('Survived',axis=1)
trn_df_Y = trn_df['Survived']
tst_df_X = tst_df


# In[200]:


df_all1.dtypes


# In[204]:


X_train, X_test, y_train, y_test = train_test_split(trn_df_X, trn_df_Y, random_state=45)


# In[205]:


X_train.head()


# In[216]:


# Scaling and Polynomial feature transformation
scaler = MinMaxScaler()
poly = PolynomialFeatures(degree=2)

# Xtrain_poly = poly.fit_transform(X_train)
# Xtest_poly = poly.transform(X_test)
# Xsubtest_poly = poly.transform(tst_df_X)
# Xtrain_poly = scaler.fit_transform(Xtrain_poly)
# Xtest_poly = scaler.transform(Xtest_poly)
# Xsubtest_poly = scaler.transform(Xsubtest_poly)

Xtrain_poly = scaler.fit_transform(X_train)
Xtest_poly = scaler.transform(X_test)
Xsubtest_poly = scaler.transform(tst_df_X)


# In[217]:


# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 300,
#                                cv = 4, verbose=2, random_state=0, n_jobs = -1, scoring='accuracy')
# # Fit the random search model
# rf_random.fit(Xtrain_poly, y_train)


# In[218]:


def evaluate(model, features, label):
    prediction = model.predict(features)
    accuracy = accuracy_score(label, prediction)
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy*100))
    print(confusion_matrix(label, prediction))
    return accuracy, prediction


# In[223]:


base_model = RandomForestClassifier(n_estimators = 220, max_depth = 5, random_state = 0)
base_model.fit(Xtrain_poly, y_train)
base_accuracy, base_pred = evaluate(base_model, Xtest_poly, y_test)


# In[ ]:


# rf_random.best_estimator_


# **BEST Random Estimator:**
# > RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=10, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=2, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)

# In[226]:


# best_random = rf_random.best_estimator_
best_random = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=220, n_jobs=None,
            oob_score=False, random_state=0, verbose=0,
            warm_start=False)
best_random.fit(Xtrain_poly, y_train)
random_accuracy, rand_pred = evaluate(best_random, Xtest_poly, y_test)


# In[227]:


y_trpred = best_random.predict(Xtrain_poly)
y_pred = best_random.predict(Xtest_poly)
y_subtest = best_random.predict(Xsubtest_poly)
print('Train data accuracy score: {:0.4f}% '.format(best_random.score(Xtrain_poly, y_train)*100))
print('Test data accuracy score: {:0.4f}% '.format(best_random.score(Xtest_poly, y_test)*100))
df_Subm['Survived'] = y_subtest.astype('int')
df_Subm.to_csv('submission_FE_RDRF.csv', index=False)
print(os.listdir("../working"))


# In[80]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 20, 30],
    'max_features': [2, 3, 'auto'],
    'min_samples_leaf': [1, 2, 3, 4],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [100, 200, 220, 300, 400]
}
# Create a base model
CLF = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = CLF, param_grid = param_grid, scoring='accuracy',
                           cv = 4, n_jobs = -1, verbose = 2)


# In[81]:


#Fit the gridsearch over training data to find the best parameter
grid_search.fit(Xtrain_poly, y_train)


# In[82]:


grid_search.best_estimator_


# > RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=5, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=4,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)

# In[83]:


grid_search.best_score_


# **BEST GridSearch Estimator:**
# > RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=20, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=4, min_samples_split=6,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)

# In[212]:


# best_grid = grid_search.best_estimator_
best_grid = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=20, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
best_grid.fit(Xtrain_poly, y_train)
grid_accuracy, grid_pred = evaluate(best_grid, Xtest_poly, y_test)


# In[140]:


grid_search.best_params_


# In[213]:


y_trpred = best_grid.predict(Xtrain_poly)
y_pred = best_grid.predict(Xtest_poly)
y_subtest = best_grid.predict(Xsubtest_poly)
print('Train data accuracy score: {:0.4f}% '.format(best_grid.score(Xtrain_poly, y_train)*100))
print('Test data accuracy score: {:0.4f}% '.format(best_grid.score(Xtest_poly, y_test)*100))
df_Subm['Survived'] = y_subtest.astype('int')
df_Subm.to_csv('submission_FE_GSRF.csv', index=False)
print(os.listdir("../working"))


# In[214]:


# Fitting the training data to default XGBClassifier
xgb_model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200, verbosity=1, 
                          objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, 
                          min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, 
                          colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
                          random_state=0)
xgb_model.fit(Xtrain_poly, y_train)
_, _ = evaluate(xgb_model, Xtest_poly, y_test)


# In[215]:


y_trpredxg = xgb_model.predict(Xtrain_poly)
y_predxg = xgb_model.predict(Xtest_poly)
y_subtestxg = xgb_model.predict(Xsubtest_poly)
print('Train data accuracy score: {:0.4f}% '.format(xgb_model.score(Xtrain_poly, y_train)*100))
print('Test data accuracy score: {:0.4f}% '.format(xgb_model.score(Xtest_poly, y_test)*100))
df_Subm['Survived'] = y_subtestxg.astype('int')
df_Subm.to_csv('submission_FE_XGB.csv', index=False)
print(os.listdir("../working"))


# In[ ]:




