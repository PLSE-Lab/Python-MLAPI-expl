#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset Kernel

# This notebook is intended for anyone to get a walkthrough of how they can proceed with the Titanic dataset. I do not explain the logic behind basic Python/SKlearn code in this notebook. However, the logic behind this is provided at the end of this notebook. You are encouraged to be curious throughout this journey. If you liked this kernel, please upvote it so that it serves as an inspiration for me :) 

# ### Why is this notebook worth your time?
# 
# Kaggle has a lot of kernels which show the completed product, but couldn't find one which takes you through the journey on how steps such as feature engineering and grid search will be able to help a data scientist along the way. In this notebook, I iterate over the problem and try to figure out solutions along the way, in an attempt to model how someone actually works on a dataset and improves it. The objective of this notebook is not to show you a completed notebook with complete explanations on every single topic and terminology you come across, but rather, to show you how an imperfect solution can slowly be iterated to something that is better than the previous solution and to make you curious about concepts that you perhaps might not have come across before

# ## Table of Contents
# 
# 1. [Import statements](#Import-statements)
# 2. [Exploratory Data Analysis(EDA)](#Exploratory-Data-Analysis(EDA)
# 3. [Imputing & Cleaning](#Imputing-&-Cleaning)
# 4. [Modelling](#Modelling)
# 5. [Concluding Notes](#Concluding-Notes)

# ### Import statements

# Lets start out by importing all required libraries

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb


# Lets disable warnings so that we can keep this notebook clean

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head(10)


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# In[ ]:


train_df.dtypes


# ### Exploratory Data Analysis(EDA)

# In[ ]:


train_df.isna().sum()


# In[ ]:


train_df['Cabin'].nunique()


# Compared to the fact that we have 890 rows of data, 150 unique values is very high and might not lead to much of an improvement in our modelling process, let's drop this later

# In[ ]:


train_df['Embarked'].unique()


# In[ ]:


train_df['Embarked'].nunique()


# The *Embarked* attribute only has 3 unique values, so let's keep this feature in our dataset

# In[ ]:


sns.distplot(train_df['Age'].dropna())


# ### Imputing & Cleaning

# In this section, we are going to clean the data by removing unnecessary features and imputing values for missing data

# In[ ]:


train_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
train_df = train_df[train_df.Embarked.notna()]
train_df.head()


# In[ ]:


passengerList = test_df['PassengerId']
test_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
test_df.head()


# In[ ]:


X = train_df.loc[:, train_df.columns != 'Survived']
y = train_df.loc[:, 'Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y)


# As we saw in the distribution plot above, *Age* is a skewed feature. To impute skewed features, we usually use the Median of that feature. If it was not skewed, then we would be using the Mean of the feature in most cases.

# In[ ]:


imputer = SimpleImputer(strategy='median')
imputer.fit(X_train['Age'].to_numpy().reshape(-1, 1))

X_train['Age'] = imputer.transform(X_train['Age'].to_numpy().reshape(-1, 1))
X_test['Age'] = imputer.transform(X_test['Age'].to_numpy().reshape(-1, 1))

test_df['Age'] = imputer.transform(test_df['Age'].to_numpy().reshape(-1, 1))


# In[ ]:


X_train['Age'] = X_train['Age'].round()
X_test['Age'] = X_test['Age'].round()

X_train['Fare'] = X_train['Fare'].round()
X_test['Fare'] = X_test['Fare'].round()

test_df['Age'] = test_df['Age'].round()
test_df['Fare'] = test_df['Age'].round()


# The *pd.get_dummies* is used to split out the categorical variables and performs [One-Hot Encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) on the data. 

# In[ ]:


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
test_df = pd.get_dummies(test_df)


# ### Modelling

# I feel that our data is now in a good position for us to go ahead and create models out of. Lets start by using Random Forest.

# **Note:** When you run the below cell block, you will get a different accuracy and that's completely alright. Your dataset will be split in a different way and I am not using random state here to get unified results as I would like the reader to execute these and learn how these accuracies differ with the steps we perform

# In[ ]:


gridParameters = {'n_estimators': [1, 5, 10, 50, 100],
                 'max_depth': [None, 1, 5, 10, 50, 100]}

model = RandomForestClassifier()
clf = GridSearchCV(model, gridParameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# **This solution has given me an accuracy of 78% on the validation set. Again, please note that when you run this block, your accuracy will be different due to the above mentioned reason**

# We are now going to train multiple classification algorithms on the same dataset and then extract the classifier that provided the best accuracy metrics

# In[ ]:


#Function to run different classifications algorithms. Returns the clf object of the classifier that gave highest accuracy

def getBestClassifier(X_train, y_train, X_test, y_test):    
    classifierList = {
        'SVM': SVC(),
        'Neural Network': MLPClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    classifierParams = {
        'SVM': {'C': [0.01, 0.1, 1, 10, 100],
               'kernel': ['linear', 'rbf', 'sigmoid']},
        'Neural Network': {'activation': ['identity', 'logistic', 'tanh', 'relu']},
        'Random Forest': {'n_estimators': [1, 5, 10, 50, 100],
                     'max_depth': [None, 1, 5, 10, 50, 100]}
    }
    
    fittedClassifiersParam = {}
    
    for key, classifier in classifierList.items():
        clf = GridSearchCV(classifier, classifierParams[key])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fittedClassifiersParam[key] = [accuracy_score(y_test, y_pred), clf.best_estimator_]
        print('Accuracy of {0:20s}: {1}'.format(key, str(accuracy_score(y_test, y_pred))))
    
    return fittedClassifiersParam[sorted(fittedClassifiersParam, key = lambda k : fittedClassifiersParam[k][0], reverse=True)[0]]


# In[ ]:


#Used to plot a confusion matrix

def confusionMatrix(y_test, y_pred):
    fig, ax = plt.subplots()
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat, annot = True, fmt='d')

    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    ax.set_ylim([0, 2])


# In[ ]:


bestClassifier = getBestClassifier(X_train, y_train, X_test, y_test)


# In[ ]:


bestClassifier


# In[ ]:


confusionMatrix(y_test, bestClassifier[1].predict(X_test))


# **After performing a grid search, the best model that we have is a Neural Network which is predicting at an accuracy of 77.1%. Neural Network models do not have feature importances and you will have to use different techniques to understand feature importance. However, if you got another classifier with better parameters, you would be able to seee a graph below, showing the importance of all the features in the model**

# In[ ]:


if bestClassifier[1].__class__.__name__ == 'MLPClassifier':
    print("Feature Importance not available for the model chosen: " + str(bestClassifier[1].__class__.__name__) )
else:
    plt.figure(figsize =(12, 6))
    plt.title(bestClassifier[1].__class__.__name__)
    sns.barplot(x=X_train.columns, y=bestClassifier[1].feature_importances_)
    plt.xticks(rotation=45, horizontalalignment='right')


# In[ ]:


X_train['MemCount'] =  X_train['SibSp'] + X_train['Parch']
X_test['MemCount'] =  X_test['SibSp'] + X_test['Parch']
test_df['MemCount'] =  test_df['SibSp'] + test_df['Parch']

X_train.drop(['SibSp', 'Parch'], inplace=True, axis=1)
X_test.drop(['SibSp', 'Parch'], inplace=True, axis=1)
test_df.drop(['SibSp', 'Parch'], inplace=True, axis=1)

X_train['isAlone'] = X_train['MemCount'].apply(lambda x: 1 if x > 0 else 0)
X_test['isAlone'] = X_test['MemCount'].apply(lambda x: 1 if x > 0 else 0)
test_df['isAlone'] = test_df['MemCount'].apply(lambda x: 1 if x > 0 else 0)

X_train['Age'] = pd.cut(X_train['Age'], 4, labels=[1, 2, 3, 4])
X_test['Age'] = pd.cut(X_test['Age'], 4, labels=[1, 2, 3, 4])
test_df['Age'] = pd.cut(test_df['Age'], 4, labels=[1, 2, 3, 4])

X_train['Fare'] = pd.cut(X_train['Fare'], 4, labels=[1, 2, 3, 4])
X_test['Fare'] = pd.cut(X_test['Fare'], 4, labels=[1, 2, 3, 4])
test_df['Fare'] = pd.cut(test_df['Fare'], 4, labels=[1, 2, 3, 4])


# In[ ]:


X_train['Age'] = X_train['Age'].astype('int')
X_train['Fare'] = X_train['Fare'].astype('int')

X_test['Age'] = X_test['Age'].astype('int')
X_test['Fare'] = X_test['Fare'].astype('int')

test_df['Fare'] = test_df['Fare'].astype('int')
test_df['Age'] = test_df['Age'].astype('int')


# In[ ]:


bestClassifier = getBestClassifier(X_train, y_train, X_test, y_test)


# After doing a bit of feature engineering, you can see that Random Forest is now performing better and giving better predictions.

# In[ ]:


confusionMatrix(y_test, bestClassifier[1].predict(X_test))


# In[ ]:


if bestClassifier[1].__class__.__name__ in ['MLPClassifier', 'SVC']:
    print("Feature Importance not available for the model chosen: " + str(bestClassifier[1].__class__.__name__) )
else:
    plt.figure(figsize =(12, 6))
    sns.barplot(x=X_train.columns, y=bestClassifier[1].feature_importances_)
    plt.xticks(rotation=45, horizontalalignment='right')


# To try to increase the accuracy further, we are going to bring about the following changes:
# 
# * In the feature engineering side, let's bring back the name column and use only the title of the names. Using the name column will give us access to a plethora of new indirect information such as the status of the person
# * In the modelling side, we are going to bring use a few gradient boosted algorithms, namely, XGBoost, Catboost & LightGBM. We are then going to create an ensemble which will comprise of Neural Networks and Gradient Boosted trees

# To bring back the title column, we have to read the information again and then extract the titles from the *Name* column

# In[ ]:


titleTrainDf = pd.read_csv('../input/titanic/train.csv')
titleTestDf = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


titleTrainDf = titleTrainDf.filter(['Name'])
titleTestDf = titleTestDf.filter(['Name'])


# In[ ]:


trainTitleSeries = titleTrainDf['Name'].str.split(", ").apply(lambda x: x[1]).str.split(".").apply(lambda x : x[0])
trainTitleSeries = trainTitleSeries.rename('Title')

testTitleSeries = titleTestDf['Name'].str.split(", ").apply(lambda x: x[1]).str.split(".").apply(lambda x : x[0])
testTitleSeries = testTitleSeries.rename('Title')


# In[ ]:


trainTitleSeries.value_counts()


# As we can see above, a lot of Titles occur only once in the dataset. Let us group them all into a title called *Rare*

# In[ ]:


# Function to map the names to appropriate titles

def classifyTitles(x):
    return 'Rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x


# In[ ]:


trainTitleSeries = trainTitleSeries.apply(classifyTitles)
testTitleSeries = testTitleSeries.apply(classifyTitles)


# In[ ]:


X_train = X_train.merge(trainTitleSeries, left_index=True, right_index=True)
X_test = X_test.merge(trainTitleSeries, left_index=True, right_index=True)

test_df = test_df.merge(testTitleSeries, left_index=True, right_index=True)


# In[ ]:


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
test_df = pd.get_dummies(test_df)


# * **classiferList:** Contains list of names algorithms we are going to use and its corresponding objects
# * **classifierParams:** Contains list of names of algorithms we are going to use and its viable parameters. We are going to perform Grid Search on these parameters

# In[ ]:


classifierList = {
    'SVM': SVC(),
    'Neural Network': MLPClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(silent=1, verbose_eval=False),
    'CatBoost': CatBoostClassifier(logging_level='Silent'),
    'LightGBM': lgb.LGBMClassifier()
}

gridEstimatorCount = [1, 5, 10, 50, 100]
gridMaxDepth = [1, 2, 4, 5, 8, 10]
gridLearningRate = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]

classifierParams = {
    'SVM': {'C': [0.01, 0.1, 1, 10, 100],
           'kernel': ['linear', 'rbf', 'sigmoid']},
    'Neural Network': {'activation': ['identity', 'logistic', 'tanh', 'relu']},
    'Random Forest': {'n_estimators': gridEstimatorCount,
                 'max_depth': gridMaxDepth},
    'XGBoost': {'learning_rate': gridLearningRate, 
            'max_depth': gridMaxDepth, 
            'n_estimators': gridEstimatorCount},
    'CatBoost': {'n_estimators': gridEstimatorCount,
                'max_depth': gridMaxDepth},
    'LightGBM': {'n_estimators': gridEstimatorCount,
                'max_depth': gridMaxDepth}
}


# In[ ]:


# Create an ensemble and return the ensemble object

def createEnsemble(X_train, y_train, X_test=[], y_test=[], classifierList={}, isFullDataset=False):        
    fittedClassifiers = {}
    
    if not classifierList:
        return
    
    for key, classifier in classifierList.items():
        
        print("Now training: ", key)
        
        clf = GridSearchCV(classifier, classifierParams[key], cv=5, n_jobs=-1, scoring='accuracy')
        clf.fit(X_train, y_train)
        fittedClassifiers.update({key: clf.best_estimator_})

        if not isFullDataset:
            y_pred = clf.predict(X_test)
            print(key + ' has accuracy: ' + str(accuracy_score(y_test, y_pred)))
    
    ensemble = VotingClassifier(estimators=[(k, v) for k, v in fittedClassifiers.items()])
    return ensemble


# Lets create an ensemble with Neural Network, LightGBM and Random Forest

# In[ ]:


classifierList = {
    'Neural Network': MLPClassifier(),
    'Random Forest': RandomForestClassifier(),
    'LightGBM': lgb.LGBMClassifier()
}


# In[ ]:


bestClassifier = createEnsemble(X_train, y_train, X_test, y_test, classifierList=classifierList)


# **Running model on the entire dataset. This will be submitted for prediction**

# In[ ]:


finalTrainingDataset = pd.concat([X_train, X_test])
finalTargetDataset = pd.concat([y_train, y_test])

ensemble = createEnsemble(finalTrainingDataset, finalTargetDataset, classifierList=classifierList, isFullDataset=True)

results = ensemble.fit(finalTrainingDataset, finalTargetDataset).predict(test_df)


# **Creating CSV for submission:**

# In[ ]:


pd.concat([passengerList, pd.Series(results, name='Survived')], axis=1).to_csv('080620_final_ensemble.csv', index=False)


# ### Concluding Notes
# 
# * A few things in the notebook has been left unexplained. The reason for this is that unless, you, as a reader, do not search for this information and read in detail about it, you are not going to have a proper understanding of the concept. I encourage you to google for any concept that you may be unfamiliar with.
# * If you have any suggestions on how this kernel can be improved or if you spot any mistakes, please feel free to point them out in the comments section
# * While you are executing the code blocks, you may sometimes notice that the Accuracy might have decreased when you performed some step. However, this is perfectly normal as there is no guarantee that what you are doing will help you get to the top of the leaderboard. You should strive to continue to try out new things and read more & more kernels to learn new approaches. 
