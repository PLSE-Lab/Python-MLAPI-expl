#!/usr/bin/env python
# coding: utf-8

# # Working with Data Dictionary: simplier, faster and more accurate way to understand the data.

# This notebook not for very beginners but for entry and middle-level Data Scientists. I want to share here a simple way to make your job easy and produce a more accurate result.
# 
# What you will NOT find here:
# 1. EDA
# 2. Plots
# 3. Long explanations which is called in my country a "text water"
# 
# What you will find:
# 1. Data Dictionary
# 2. Data Types explanation
# 3. My choise of feature engineering (some of features you may not expect)
# 4. Model selection
# 5. Model tuning (feature also)
# 6. Modeling
# 
# Let's get started.

# # Imports and styling.

# In[ ]:


import numpy as np
import pandas as pd
import sklearn 

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import platform
print('Versions:')
print('  python', platform.python_version())
n = ('numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn')
nn = (np, pd, sklearn, mpl, sns)
for a, b in zip(n, nn):
    print('  --', str(a), b.__version__)


# In[ ]:


#pandas styling
pd.set_option('colheader_justify', 'left')
pd.set_option('precision', 0)
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_colwidth', -1)


# In[ ]:


#pd.reset_option('all')


# In[ ]:


#seaborn syling
sns.set_style('whitegrid', { 'axes.axisbelow': True, 'axes.edgecolor': 'black', 'axes.facecolor': 'white',
        'axes.grid': True, 'axes.labelcolor': 'black', 'axes.spines.bottom': True, 'axes.spines.left': True,
        'axes.spines.right': False, 'axes.spines.top': False, 'figure.facecolor': 'white', 
        #'font.family': ['sans-serif'], 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'grid.color': 'grey', 'grid.linestyle': ':', 'image.cmap': 'rocket', 'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'text.color': 'black', 
        'xtick.top': False, 'xtick.bottom': True, 'xtick.color': 'navy', 'xtick.direction': 'out', 
        'ytick.right': False,    'ytick.left': True, 'ytick.color': 'navy', 'ytick.direction': 'out'})


# In[ ]:


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process 
from sklearn import feature_selection, model_selection, metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier


# ## Concatenating data sets

# In[ ]:


train_raw = pd.read_csv('../input/titanic/train.csv')
test_raw = pd.read_csv('../input/titanic/test.csv')
len(train_raw) + len(test_raw)


# In[ ]:


df = pd.concat(objs=[train_raw, test_raw], axis=0)
df.shape


# ## Open a Data Dictionary

# In[ ]:


#ddict = pd.read_csv('../input/Titanic_Data_Dictionary_ready.csv', index_col=0)
#ddict


# *For some reason I can't use my data here, Kaggle support team still works with my problem (I hope so) - so, I've put an image of dictionary here.* 
# 
# *But everything is working on GitHub and [this notebook you can find here](https://github.com/datalanas/Jupyter_notebooks_to_share/blob/master/Titanic_Prediction_of_binary_events.ipynb) and if it's difficult to you to create such kind of table, you can find the way I've made [this dictionary exactly ](https://github.com/datalanas/Jupyter_notebooks_to_share/blob/master/Titanic_What_is_DataDictionary.ipynb) at GitHub also.*

# ![image.png](attachment:image.png)

# Data Dictionary has main data set's feature's names as indexes and followed columns:
# - 'Definition' - meaning of feature
# - 'Description' - meaning of feature's values 
# - '#Unique' - number of unique values in the columns, where NaN calculated as value also  
# - 'TopValue' - the most used value 
# - '%UsedTop' - % of using top value
# - '%Missing' - % of missing values
# - 'Unit' - measurement units 
# - 'Type' - measurement scales                  
# - 'Dtype' - column's python data type  

# Column "Type" may has followed values:
# 1. Useless (useless for machine learning algorithms)
# 2. Nominal (truly categorical, labels or groups without order)
# 3. Binary (dichotomous - a type of nominal scales that contains only two categories)
# 4. Ordinal (groups with order)
# 5. Discrete  (count data, the number of occurrences)
# 6. Cont (continuous with an absolute zero,but without a temporal component)
# 7. Interval (continuous without an absolute zero, but without a temporal component)
# 8. Time (cyclical numbers with a temporal component; continuous)
# 9. Text
# 10. Image
# 11. Audio
# 12. Video

# Creating such kind of table takes not a lot, but save much more time and gives me feeling of TRUE understanding the data.
# Hope you feel the same.

# # Data engineering

# #### useless features

# We call feature "useless" if it:
# - has a single unique value
# - brings no information to algoritms, like "PassengerID"
# - has too many missing values, like "Cabin"
# - highly correlated
# - has a zero importance in a tree-based model

# In[ ]:


col1 = test_raw['PassengerId'] # will need for a submission
df.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)


# #### missing values

# In[ ]:


df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna('S')
df.isnull().sum().to_frame().T


# #### feature engineering

# In[ ]:


#family sizes
df['Fsize'] =  df['Parch'] + df['SibSp'] + 1


# In[ ]:


#titles
df['Surname'], df['Name'] = zip(*df['Name'].apply(lambda x: x.split(',')))
df['Title'], df['Name'] = zip(*df['Name'].apply(lambda x: x.split('.')))

titles = (df['Title'].value_counts() < 10)
df['Title'] = df['Title'].apply(lambda x: ' Misc' if titles.loc[x] == True else x)
df['Title'].value_counts().to_frame().T


# In[ ]:


# ticket set (how many person in one ticket)
# if one ticket has family members only, it's "monotinic"; if not - "mixed"
df['Tname'] = df['Ticket']
df['Tset']=0

for t in df['Tname'].unique():
    if df['Surname'].loc[(df['Tname']==t)].nunique() != 1:
        df['Tset'].loc[(df['Tname']==t)] = 'mixed'
    else: 
        df['Tset'].loc[(df['Tname']==t)] = 'monotonic'

for t in df['Tname'].unique():
    if df['Surname'].loc[(df['Tname']==t)].nunique() != 1:
        df['Tset'].loc[(df['Tname']==t)] = 'mixed'
    else: 
        df['Tset'].loc[(df['Tname']==t)] = 'monotonic'


# In[ ]:


#price and 
for t in df['Ticket'].unique():
    df['Ticket'].loc[(df['Ticket']==t)] = len(df.loc[(df['Ticket']==t)]) 
df['Price'] = df['Fare'] / df['Ticket']
#renaming "Ticket"
df.rename(columns={'Ticket':'Tgroup'}, inplace=True)


# In[ ]:


#deleting useless again
df.drop(['Parch', 'SibSp', 'Name', 'Surname', 'Tname', 'Fare'], axis=1, inplace=True)


# In[ ]:


df.head(2)


# #### Transforming

# In[ ]:


df.dtypes.to_frame().sort_values([0]).T


# In[ ]:


#code categorical data
label = LabelEncoder()
cols = df.dtypes[df.dtypes == 'object'].index.tolist()
for col in cols:
    df[col] = label.fit_transform(df[col])


# In[ ]:


#binning
df['Price'] = pd.qcut(df['Price'], 4)
df['Age'] = pd.cut(df['Age'].astype(int), 5)


# In[ ]:


#code binning data
df['Age'] = label.fit_transform(df['Age'])
df['Price'] = label.fit_transform(df['Price'])


# In[ ]:


df.head()


# ## Splitting back to train and test

# In[ ]:


a = len(train_raw)
train = df[:a]
test = df[a:]


# In[ ]:


train_raw.shape[0] == train.shape[0]


# In[ ]:


test.drop(['Survived'], axis=1, inplace=True)
test_raw.shape[0] == test.shape[0]


# # Modeling

# In[ ]:


X = train.drop(['Survived'], axis=1).columns.to_list()
y = ['Survived']


# In[ ]:


#Machine Learning Algorithm initialization
MLA = [ #Ensemble Methods
        ensemble.AdaBoostClassifier(), ensemble.BaggingClassifier(), ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(), ensemble.RandomForestClassifier(),
        #Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),
        #GLM
        linear_model.LogisticRegressionCV(), linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(), linear_model.SGDClassifier(), linear_model.Perceptron(),
        #Navies Bayes
        naive_bayes.BernoulliNB(), naive_bayes.GaussianNB(),
        #Nearest Neighbor
        neighbors.KNeighborsClassifier(),
        #SVM
        svm.SVC(probability=True), svm.NuSVC(probability=True), svm.LinearSVC(),
        #Trees    
        tree.DecisionTreeClassifier(), tree.ExtraTreeClassifier(),
        #Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(), discriminant_analysis.QuadraticDiscriminantAnalysis(),
        #xgboost
        XGBClassifier() ]


# In[ ]:


cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

mla = pd.DataFrame(columns=['Name','TestScore','ScoreTime','FitTime','Parameters'])
prediction = train[y]

i = 0
for alg in MLA:
    name = alg.__class__.__name__
    mla.loc[i, 'Name'] = name
    mla.loc[i, 'Parameters'] = str(alg.get_params())
    
    cv_results = model_selection.cross_validate(alg, train[X], train[y], cv=cv_split)
        
    mla.loc[i, 'FitTime'] = cv_results['fit_time'].mean()
    mla.loc[i, 'ScoreTime'] = cv_results['score_time'].mean()
    mla.loc[i, 'TestScore'] = cv_results['test_score'].mean()

    alg.fit(train[X], train[y])
    prediction[name] = alg.predict(train[X])    
    i += 1

mla = mla.sort_values('TestScore', ascending=False).reset_index(drop=True)
mla


# #### parameters tuning

# In[ ]:


# assing parameters
param_grid = {'criterion': ['gini', 'entropy'],  #default is gini
              #'splitter': ['best', 'random'], #default is best
              'max_depth': [2,4,6,8,10,None], #default is none
              #'min_samples_split': [2,5,10,.03,.05], #default is 2
              #'min_samples_leaf': [1,5,10,.03,.05], #default is 1
              #'max_features': [None, 'auto'], #default none or all
              'random_state': [0]}

#choose best model with grid_search
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split)
tune_model.fit(train[X], train[y])
print('Parameters: ', tune_model.best_params_)


# #### feature selection

# In[ ]:


clf = tree.DecisionTreeClassifier()
results = model_selection.cross_validate(clf, train[X], train[y], cv=cv_split)
clf.fit(train[X], train[y])
results['test_score'].mean()*100

#feature selection
fs = feature_selection.RFECV(clf, step=1, scoring='accuracy', cv=cv_split)
fs.fit(train[X], train[y])

#transform x and y to fit a new model
X = train[X].columns.values[fs.get_support()]
results = model_selection.cross_validate(clf, train[X], train[y], cv=cv_split)

print('Shape New: ', train[X].shape) 
print('Features to use: ', X)


# In[ ]:


#tune parameters
tuned = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv=cv_split)
tuned.fit(train[X], train[y])

param_grid = tuned.best_params_
param_grid


# In[ ]:


clf = ensemble.GradientBoostingClassifier()
results = model_selection.cross_validate(clf, train[X], train[y], cv=cv_split)
clf.fit(train[X], train[y])
results['test_score'].mean()*100


# ## Prediction

# In[ ]:


test['Survived'] = clf.predict(test[X])


# ## Submission

# In[ ]:


submit = pd.DataFrame({ 'PassengerId' : col1, 'Survived': test['Survived'] }).set_index('PassengerId')
submit['Survived'] = submit['Survived'].astype('int')
submit.to_csv('submission.csv')


# This simple method gives me 0.799 score in Kaggle, what is a top 12% for today.

# I will be glad to see any your questions or suggestions. Criticism welcomed also.  
# Many thanks for your time.  
# Best,  
# Lana  
