#!/usr/bin/env python
# coding: utf-8

# ##  **Titanic ML Prediction - First Kaggle Notebook**

# In[ ]:


#Titanic Tutorial | Kaggle
#just in case if we require tutorial
#https://www.kaggle.com/alexisbcook/titanic-tutorial - Kaggle tutorial
#https://www.kaggle.com/sinakhorami/titanic-best-working-classifier - One that i reffered to for data cleaning 
#https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/data - one i reffered


# ## Build a predictive model which answers following:
#  1. what sorts of people were more likely to survive? 
#  *  Data :- (name, age, gender, socio-economic class, etc). 
#  * Total Passengers = 2224
#  * Total Death toll = 1502

# ##  Basic Terminologies
# 
#  * Ensemble - Including other models in a model for better predictions.

# ##  **INFORMATIONS**
#  
#  * AdaBoostClassifier -> AdaBoost is a type of "Ensemble Learning" where multiple learners are employed to build a stronger learning algorithm. AdaBoost works by choosing a base algorithm (e.g. decision trees) and iteratively improving it by accounting for the incorrectly classified examples in the training set.
#  
#  * Gradient BoostingClassifier -> Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees
#  
#  * ExtraTreeClassifier -> ExtraTreeClassifier is an extremely randomized version of DecisionTreeClassifier meant to be used internally as part of the ExtraTreesClassifier ensemble.
#  
#  * loc -> DataFrame.loc[] method is a method that takes only index labels and returns row or dataframe if the index label exists in the caller data frame.
#  
#  * pd.qcut -> The simplest use of qcut is to define the number of quantiles and let pandas figure out how to divide up the data.
#  
#  * pandas.crosstab -> Compute a simple cross tabulation of two (or more) factors. By default computes a frequency table of the factors unless an array of values and an aggregation function are passed.

# ## STATISTICAL INFORMATION
# 
#  **Standard deviation -> Standard deviation is a number used to tell how measurements for a group are spread out from the average (mean), or expected value. A low standard deviation means that most of the numbers are close to the average. A high standard deviation means that the numbers are more spread out.**

# ## ** Libraries Imported and it's real uses.**
# * sklearn -> It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. 
#  * re -> to import regular expression.
#  * xgboost ->  XGBoost stands for eXtreme Gradient Boosting and is based on decision trees.It's main goal is to push the extreme of the computation limits of machines to provide a scalable, portable and accurate for large scale tree boosting.
#  * seaborn -> Seaborn is a Python data visualization library with an emphasis on statistical plots. The library is an excellent resource for common regression and distribution plots, but where Seaborn really shines is in its ability to visualize many different features at once
#  * matplotlib.pyplot -> Matplotlib is a Python plotting library which helps you to create visualization of the data in 2 -D graph. 
#  * %matplotlib inline -> It is an IPython-specific directive which causes IPython to display matplotlib plots in a notebook cell rather than in another window. To run the code as a script use.
#  * plotly -> The plotly Python library (plotly.py) is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.
#  * plotly.offline -> No need to call the plotly server again and again for plotting.
#  * plotly.graph_objs -> Import the graph objects and elements .Example - from plotly.graph_objs import Scatter, Layout, Data, Figure
#  * sklearn.ensemble - The sklearn.ensemble module includes ensemble-based methods for classification, regression and anomaly detection.
#  
# 

# In[ ]:


import numpy as np
import pandas as pd
import re #regular expression 
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore') #ignore the warnings.

from sklearn.ensemble import (RandomForestClassifier , AdaBoostClassifier , 
                             GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC # from support vector machine import support vector classifier
from sklearn.model_selection import KFold


# In[ ]:


# Load in the train and test datasets
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# ## FEATURE ENGINEERING RETHINK AND REBUILD

# In[ ]:


full_data = [train, test]

# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;


# In[ ]:


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], ax is = 1)
test  = test.drop(drop_elements, axis = 1)


# ## Visualisations

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# ## Pairplots

# In[ ]:


g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


# ## Ensembling and Stacking module

# In[ ]:


ntrain = train.shape[0] # gives total number of training datas
ntest = train.shape[0] #gives total number of testing data
SEED = 0 # for same random distribution of data for reproducibility
NFOLDS = 5 # to fold a record using kfold
kf = KFold(ntrain , n_folds = NFOLDS , random_state = SEED)


# **def init** : Python standard for invoking the default constructor for the class. This means that when you want to create an object (classifier), you have to give it the parameters of clf (what sklearn classifier you want), seed (random seed) and params (parameters for the classifiers).
# 
# The rest of the code are simply methods of the class which simply call the corresponding methods already existing within the sklearn classifiers. Essentially, we have created a wrapper class to extend the various Sklearn classifiers so that this should help us reduce having to write the same code over and over when we implement multiple learners to our stacker.

# In[ ]:


class SKlearnHelper(object):
    def __init__(self,clf,seed = 0 ,params = None):
        params['random_state'] = seed
        self.clf = clf(**params)#keyword arguments
    def train(self, x_train , y_train):
        self.clf.fit(x_train , y_train)
    def predict(self , x):
        return self.clf.predict(x)
    def fit(self,x,y):
        return self.clf.fit(x,y)
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        


# **Out-of-Fold Predictions**
# *  stacking uses predictions of base classifiers as input for training to a second-level model.
# *   However one cannot simply train the base models on the full training data, generate predictions on the full test set and then output these for the second-level training. This runs the risk of your base model predictions already having "seen" the test set and therefore overfitting when feeding these predictions.
# 
# 

# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


np.empty((5))


# In[ ]:


#def get_oof(clf , x_train , y_train , x_test):
    oof_train = np.zeros((ntrain ,)) #np.zeros converts into array of zeros , ntrain is the total number of records in the table.
    oof_test = np.zeros((ntest ,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    print(oof_train)


# ## **LEARNINGS**
#  * A dataset should be clean for machine learning and data analysis - we can use pandas .fillna to fill null values or missing values should be replaced for analysis.
