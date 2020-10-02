#!/usr/bin/env python
# coding: utf-8

# 
# 
# # <center> Titanic </center>
# 
# 
# 
# 
# 
# 
# ### Group Number: 8
# ### Group members:
# - Abdulrahman ALQannas
# - Doaa Alsenani
# - Ghida Qahtan
# - Moayad Magadmi
# ---
# 
# 

# ## Introduction

# - In this Kaggle competition, we aim to predict which passengers survived the Titanic shipwreck according to economic status (class), sex, age .
# 
# - In this competition, we face in binary classification problem and we try to solve this problem by using:-
# 
#      - Random Forest Classifier.	
#      - KNeighbors Classifier.
#      - Support Vector Classification.
#      - Gaussian Process Classifier.
#      - Decision Tree Classifier.
#      - AdaBoost Classifier.
#      - ExtraTreesClassifier 
#      - Logistic Regression

# - ### These datasets include 11 explanatory variables:

# Train data have Survived (dependent variable) and other predictor variables.
# Test data include the same variables that in train data, but without Survived (dependent variable) because this data will be submitted to kaggle.

# -  #### Data Dictionary
# 
# |Feature|Dataset|Description|
# |-------|---|---|
# |Survival|Train|The number of survived the Titanic shipwreck| 
# |Pclass|Train/Test|Economic status (class)| 
# |Sex|Train/Test|male or female.| 
# |Age|Train/Test|Age in years| 
# |Sibsp/Parch|Train/Test|The number of siblings, spouses, or children aboard the Titanic.| 
# |ticket|Train/Test|Ticket number.| 
# |Fare|Train/Test|Passenger fare| 
# |Cabin|Train/Test|Cabin number| 
# |Embarked|Train/Test|Port of Embarkation| 
# 
# 

# ## Importing packages

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectFromModel

# To ignore unwanted warnings
import warnings
warnings.filterwarnings('ignore')


# ## Loading the Titanic

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')


# ## Exploring the Data

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()
print('_'*40)
test.info()


# ### Check Missing Values

# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))
# Train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')
# Test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');


# In[ ]:


#missing amount for train set
missing= train.isnull().sum().sort_values(ascending=False)
percentage = (train.isnull().sum()/ train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])
missing_data.head(3)


# In[ ]:


#missing amount for test set
missing= test.isnull().sum().sort_values(ascending=False)
percentage = (test.isnull().sum()/ test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])
missing_data.head(3)


#  #### Filling A Few Missing Values
#  - #### Embarked Feature  in train dataset 

# In[ ]:


train.Embarked.fillna(value='S', inplace=True)


# In[ ]:


train['Embarked'].value_counts()


#    - #### The Pclass of missing fare in test dataset

# In[ ]:


isn = pd.isnull(test['Fare'])
test[isn]


# - #### Filling missing Fare values in test dataset 

# In[ ]:


average_of_fare= test.groupby('Pclass')['Fare'].mean()
print('The mean fare for the Pclass (for missing fare data) is:',average_of_fare[3])


# In[ ]:


# filling the missing by mean
test.Fare.fillna(value=average_of_fare[3], inplace=True)


# #### Let's see how to teat the Age column !

# **The mean age of each Pclass in the train data.**

# In[ ]:


mean_age = train.groupby('Pclass')[['Age']].mean()
mean_age


# #### We fill the mean age with respect to each Pclass.

# In[ ]:


#defining a function 'impute_age'
def impute_age(age_pclass): # passing age_pclass as ['Age', 'Pclass']
    # Passing age_pclass[0] which is 'Age' to variable 'Age'
    Age = age_pclass[0]
    # Passing age_pclass[2] which is 'Pclass' to variable 'Pclass'
    Pclass = age_pclass[1]
    #applying condition based on the Age and filling the missing data respectively 
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age


# In[ ]:


#train data
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
#test data
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)


# ### Feature Engineering

#  - ##### Cabin Feature

# In[ ]:


# train
train['Cabin']=train['Cabin'].notnull().astype('int')
train['Cabin'].unique()


# In[ ]:


# test
test['Cabin']=test['Cabin'].notnull().astype('int')
test['Cabin'].unique()


#  - ##### Age Feature

# In[ ]:


# Sex & Age
g = sns.FacetGrid(train, hue = 'Survived', col = 'Sex', height = 3, aspect = 2)
g.map(plt.hist, 'Age', alpha = .5, bins = 20)
g.add_legend()
plt.show()


# - The graph shows that the death rate of males was higher than females
# - The graph shows that older passengers had less chance of survival.

# In[ ]:


#Change the data types
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)


# In[ ]:


def age_range(df):
    df['Age'].loc[df['Age'] <= 16 ] = 0
    df['Age'].loc[(df['Age'] > 16) & (df['Age'] <= 32)] = 1
    df['Age'].loc[(df['Age'] > 32) & (df['Age'] <= 48)] = 2
    df['Age'].loc[(df['Age'] > 48) & (df['Age'] <= 64)] = 3
    df['Age'].loc[df['Age'] > 64] = 4   
age_range(train)
age_range(test)


# ### Making several new features 

# - ####  Title Feature

# In[ ]:


# Creating title dictionary in train data
titles = set()
for name in train['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())  
# Mapping Titles
train['Title'] = train.Title.map(Title_Dictionary)


# In[ ]:


# Creating Title dictionary in test data
titles = set()
for name in test['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
Title_Dictionary_test = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())    
# Mapping Titles
test['Title'] = test.Title.map(Title_Dictionary_test)


# In[ ]:


# Missing values
test[test['Title'].isnull()]


# In[ ]:


# Filling missing values in title
test['Title'].fillna(value='Mr', inplace=True)


# - #### Family Size Features

# In[ ]:


test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1


# #### No more missing data

# In[ ]:


train['FamilySize'] = train['FamilySize'].astype(int)
test['FamilySize'] = test['FamilySize'].astype(int)
def family_range(df):
    df['FamilySize'].loc[df['FamilySize'] <= 1 ] = 0
    df['FamilySize'].loc[(df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)] = 1
    df['FamilySize'].loc[df['FamilySize'] >= 5] = 2   
family_range(train)
family_range(test)


# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))
# Train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')
# Test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');


# ### Dummies
# ##### Creating Dummies For Categorical Columns.

# In[ ]:


# Train Data
train = pd.get_dummies(train, columns=['Sex','Embarked','Title'],drop_first=True)


# In[ ]:


# Test Data
test= pd.get_dummies(test, columns=['Sex','Embarked','Title'],drop_first=True)
test['Title_Royalty'] = 0    # adding Title_Royalty column to match columns in the train df


# ### Analyze by visualizing data

# - #### Survived Correlation Matrix 

# Now let's take a look at the most important variables, which will have strong linear releationship with 
# <b>Survived</b> variable .<br><br>

# In[ ]:


fig=plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(train.corr(), annot=True,ax=ax, cmap=plt.cm.YlGnBu)
ax.set_title('The correlations between all numeric features')
palette =sns.diverging_palette(80, 110, n=146)
plt.show


# In[ ]:


# correlation with the target
corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# In[ ]:


g = sns.factorplot('Survived',data=train,kind='count',hue='Pclass')
g._legend.set_title('Pclass')
# replace labels
new_labels = ['1st class', '2nd class', '3rd class']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)


# - The graph shows that the survival rate in the  3rd class was lowest than the 1st and 2nd class.

# In[ ]:


g = sns.factorplot('Pclass',data=train,hue='Sex_male',kind='count')
g._legend.set_title('Sex')
# replace labels
new_labels = ['Female', 'Male']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)


# - The graph shows that the reason for the high death rate of men than women because most of them were in the 3rd class.  

# In[ ]:


g = sns.factorplot('Survived',data=train,kind='count',hue='FamilySize')
g._legend.set_title('Family Size')
# replace labels
new_labels = ['Small', 'Single', 'Large']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)


# - The graph shows that the number of deaths in singles was more than the families due to the plan followed in the rescue, which depends on the rescue of families first and then singles. 

# ### Modeling 

# #### Model Prep: Create X  and y variables

# - #### Dropping Some Columns

# In[ ]:


# Train data
features_drop = ['PassengerId','Name', 'Ticket', 'Survived','SibSp','Parch']


# In[ ]:


selected_features = [x for x in train.columns if x not in features_drop]


# In[ ]:


# Test data
features_drop_test = ['PassengerId','Name', 'Ticket','SibSp','Parch']


# In[ ]:


selected_features_test = [x for x in test.columns if x not in features_drop_test]


# - #### Now, separate the selected column in X_train and Survived in y_train

# In[ ]:


# Train data
X = train[selected_features]
y = train['Survived']


# In[ ]:


# Test data
testing = test[selected_features_test]


# - #### Splitting and Standardizing Train Data to Obtain Test Scores

# In[ ]:


ss = StandardScaler()
Xs =ss.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3,random_state=55, stratify=y) 


# ### 1- Build Decision Tree Classifier Model

# In[ ]:


tree= DecisionTreeClassifier()
tree.fit(X_train, y_train)
print('test score' , tree.score(X_train, y_train))
print('test score' , tree.score(X_test, y_test))


# In[ ]:


y_pred =tree.predict(testing)


#  - #### Fit a BaggingClassifier with a decision tree base estimator

# In[ ]:


dt = DecisionTreeClassifier()
dt_en = BaggingClassifier(base_estimator=dt, n_estimators=100, max_features=10)
dt_en.fit(X_train, y_train)
print('test score' , dt_en.score(X_train, y_train))
print('test score' , dt_en.score(X_test, y_test))


# In[ ]:


y_pred = dt_en.predict(testing) 


# - #### Grid Search for Bagging Classifiers

# In[ ]:


param = { 'max_features': [0.3, 0.6, 1],
        'n_estimators': [50, 150, 200], 
         'base_estimator__max_depth': [3, 5, 20]}


# In[ ]:


model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True)
model_gs = GridSearchCV(model,param, cv=6, verbose=1, n_jobs=-1 )
model_gs.fit(X_train, y_train)


# In[ ]:


model_gs.best_params_


# In[ ]:


model_gs.best_estimator_.oob_score_


# ### 2 - Build Random Forest Classifier Model

# In[ ]:


randomF = RandomForestClassifier(max_depth=350, n_estimators=9, max_features=11, random_state=14, min_samples_split=3)
randomF.fit(X_train, y_train)
print('Train score :',randomF.score(X_train, y_train))
print('Ttest score :',randomF.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(randomF, X, y, cv=cv).mean()


# In[ ]:


y_pred=randomF.predict(testing)


# ### 3- Build Extra Trees Classifier Model

# In[ ]:


et = ExtraTreesClassifier(n_estimators=66, min_samples_split=7)
et.fit(X_train, y_train)
print('Train score :',et.score(X_train, y_train))
print('Ttest score :',et.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(et, X, y, cv=cv).mean()


# In[ ]:


y_pred =et.predict(testing)


# ### 4 - Build KNeighbors Classifier	Model

# In[ ]:


knn_classifier = KNeighborsClassifier(n_neighbors=7, leaf_size=48, weights='uniform',p=1)  
knn_classifier.fit(X_train, y_train)
print(knn_classifier.score(X_train, y_train))
print (knn_classifier.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(knn_classifier, X, y, cv=cv).mean()


# In[ ]:


y_pred = knn_classifier.predict(testing) 


# - #### Fit a BaggingClassifier with a Knn base estimator

# In[ ]:


knn = KNeighborsClassifier()
knn_en = BaggingClassifier(base_estimator=knn, n_estimators=45, oob_score=True, max_features=9, random_state=99)
knn_en.fit(X_train, y_train)

print(knn_en.score(X_train, y_train))
print(knn_en.score(X_test, y_test))


# In[ ]:


y_pred = knn_en.predict(testing) 


# In[ ]:


knn_en.estimators_[12]


# ### 5 - Build SVM Model

# - #### SVM with Linear

# In[ ]:


svm_l = svm.SVC(kernel='linear', C=33)
svm_l.fit(X_train, y_train)
print('Train : ', svm_l.score(X_train, y_train))
print('Test: ', svm_l.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_l, Xs, y, cv=cv).mean()


# In[ ]:


cross_val_score(randomF, X, y, cv=cv)


# In[ ]:


y_pred = svm_l.predict(testing) 


# - #### SVM with Poly

# In[ ]:


svm_p = svm.SVC(kernel='poly', C=3)
svm_p.fit(X_train, y_train)
print(svm_p.score(X_train, y_train))
print(svm_p.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_p, Xs, y, cv=cv).mean()


# In[ ]:


y_pred = svm_p.predict(testing) 


# - #### SVM with Rbf

# In[ ]:


svm_rbf = svm.SVC(kernel='rbf', C=4)
svm_rbf.fit(X_train, y_train)
print(svm_rbf.score(X_train, y_train))
print(svm_rbf.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_rbf, Xs, y, cv=cv).mean()


# In[ ]:


y_pred = svm_rbf.predict(testing) 


# ### 6- Build Logistic Regression Model
# 

# In[ ]:


logreg = LogisticRegression(max_iter=300)
logreg.fit(X_train, y_train)
print('train score' , logreg.score(X_train, y_train))
print('test score' , logreg.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(logreg, X, y, cv=cv).mean()


# In[ ]:


cross_val_score(randomF, X, y, cv=cv)


# In[ ]:


y_pred = logreg.predict(testing) 


# ### 7- Build AdaBoost Classifier Model

# In[ ]:


adaboost = AdaBoostClassifier(n_estimators=67)
adaboost.fit(X_train, y_train)
print('Train accuracy:', adaboost.score(X_train, y_train))
print('Test accuracy:',adaboost.score(X_test, y_test))


# In[ ]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(adaboost, X, y, cv=cv).mean()


# In[ ]:


y_pred = adaboost.predict(testing) 


# ### Submission

# In[ ]:


thesubmission = gender_submission.copy()
thesubmission['Survived'] = y_pred
thesubmission['Survived'].head()
thesubmission.to_csv('thesubmission.csv', index=False)


# ##  Results

# In[ ]:


list_of_Scores = list()


# In[ ]:


# Decision Tree Classifier
results = {'Model':'Decision Tree Classifier',
           'Train Score':tree.score(X_train, y_train),
           'Test Score':tree.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# Bagging Classifier with Decision Tree 
results = {'Model':'Bagging with Decision Tree ',
           'Train Score':dt_en.score(X_train, y_train),
           'Test Score':dt_en.score(X_test, y_test),
           'Kaggle Score':0.75598}
list_of_Scores.append(results)

# Random Forest Classifier
results = {'Model':'Random Forest Classifier',
           'Train Score': randomF.score(X_train, y_train),
           'Test Score':randomF.score(X_test, y_test),
           'Kaggle Score':0.77990
}
list_of_Scores.append(results)

# Extra Trees Classifier
results = {'Model':'Extra Trees Classifier',
           'Train Score':et.score(X_train, y_train),
           'Test Score': et.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# KNeighbors Classifier
results = {'Model':'KNeighbors Classifier',
           'Train Score':knn_classifier.score(X_train, y_train),
           'Test Score':knn_classifier.score(X_test, y_test),
           'Kaggle Score':0.77511}
list_of_Scores.append(results)

# Bagging Classifier with a Knn 
results = {'Model':'Bagging Classifier with Knn ',
           'Train Score': knn_en.score(X_train, y_train),
           'Test Score':knn_en.score(X_test, y_test),
           'Kaggle Score':0.66507}
list_of_Scores.append(results)

# SVM with Linear
results = {'Model':'SVM with Linear',
           'Train Score': svm_l.score(X_train, y_train),
           'Test Score':svm_l.score(X_test, y_test),
           'Kaggle Score':0.80382}
list_of_Scores.append(results)


# SVM with Poly
results = {'Model':'SVM with Poly',
           'Train Score':svm_p.score(X_train, y_train),
           'Test Score':svm_p.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# SVM with Rbf 
results = {'Model':"SVM with Rbf",
           'Train Score':svm_rbf.score(X_train, y_train),
           'Test Score':svm_rbf.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results) 


# Logistic Regression
results = {'Model':'Logistic Regression',
           'Train Score':logreg.score(X_train, y_train),
           'Test Score':logreg.score(X_test, y_test),
           'Kaggle Score':0.80382}
list_of_Scores.append(results)

# AdaBoost Classifier
results = {'Model':'AdaBoost Classifier ',
           'Train Score':adaboost.score(X_train, y_train),
           'Test Score':adaboost.score(X_test, y_test),
           'Kaggle Score':0.77511}
list_of_Scores.append(results)


# In[ ]:


df_results = pd.DataFrame(list_of_Scores)


# - #### This table provides all the scores that we got from each model.

# In[ ]:


df_results


# # Evaluation

# In this modeling we use cross-validation to evaluate the results after data cleaning. According to the Logistic Regression and SVM important featrues, we inference those featrues can play a major part in prediction. The most important featrues are: Fare,Title_Mr and they gave us a good predect of Survived feature. According to the two models's important featrues, we inference those featrues can play a major part in prediction. As we got these result:
# 
# Logistic Regression resulte:
# 
#         Train Score: 0.8154093097913323
#         Test  Score: 0.8619402985074627
# 
# SVM resulte:
# 
#         Train Score: 0.8234349919743178
#         Test  Score: 0.8544776119402985
# 
# And when we tested the Corss Validation of Logestic, the results were:
# 
# Logistic Regression resulte:
# 
#         "[0.79329609, 0.80898876, 0.83146067, 0.82022472, 0.85393258]"
# 
# SVM resulte:
# 
#         "[0.79329609, 0.80898876, 0.83146067, 0.82022472, 0.85393258]"
#         
#         
# With an average of: "0.8215805661917017" ~ 0.82 of Logistic Regression and "0.8316740945326722" ~ 0.83 of SVM this is a good ratio, as it means that the model can generalize any new data that can enter the model at 82-83 percent of accuracy, as this result indicates that the model is right fit because a low  viariance of it.
# 
# In the picture below the first three points score we get in this  modling in kaggle which it show the same public score for the two model a Logistic Regression and SVM.

# ### Kaggle Score

# ![kaggle_score.png](attachment:kaggle_score.png)
