#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

sns.set(style='white')


# ### Loading the Data and basic check

# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
test_passenger_id = test_data['PassengerId'] ## For the Submission purpose


# In[ ]:


train_len = len(train_data)
dataset = pd.concat([train_data, test_data] , axis=0)


# In[ ]:


dataset.head()


# In[ ]:


dataset.describe()


# ### Dropping the PassengerId as it is not a feature

# In[ ]:


train_data = train_data.drop('PassengerId',axis=1)


# In[ ]:


train_data.isnull().sum()


# The Age and Cabin columns have many null values and needs to be imputed.

# ### Check whether the dataset is balanced or not

# In[ ]:


plt.figure(figsize = (8,5))
sns.countplot(x=train_data['Survived'])


# The dataset is not balanced but fine for training on it

# ## Feature Analysis

# ### Pclass

# In[ ]:


plt.figure(figsize = (10,6))
sns.countplot(x=train_data['Survived'],hue='Pclass' , data=train_data, hue_order=[1,2,3])


# In[ ]:


plt.figure(figsize = (10,6))
sns.barplot(x  = 'Pclass' , y='Survived', data=train_data)


# It's clear that the Passengers in the 1st class the first priority ones during Evacuation, then the 2nd and at last the 3rd class.

# ### Sex 

# In[ ]:


plt.figure(figsize = (10,6))
sns.countplot(x=train_data['Survived'],hue='Sex' , data=train_data)


# The data shows that female had higher probability for survival 

# ### Age

# In[ ]:


plt.figure(figsize = (12,6))
sns.kdeplot(train_data[train_data['Survived'] == 1]['Age'], color = 'Blue' , label = 'Survived')
sns.kdeplot(train_data[train_data['Survived'] == 0]['Age'], color = 'Red', label = 'Not Survived' ) 
plt.xlabel('Age')
plt.ylabel('Survival Probability')
plt.grid()


# Graph shows that the young children(between 20 to 40) and infants(0-5) had a probabilty of surviving whereas the very old people had very less chances of surviving 

# ### Sibsp

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Survived', hue = 'SibSp', data=train_data)
plt.legend(bbox_to_anchor=[0.20,0,1,1])


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x='SibSp', y = 'Survived', data=train_data)


# The data shows that the people with more siblings have lower survival rate. This can be because they would have got stuck searching for their siblings.

# ### Parch

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Survived', hue = 'Parch', data=train_data)
plt.legend(bbox_to_anchor=[0.20,0,1,1])


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = 'Parch', y='Survived', data=train_data)


# ### Fare

# In[ ]:


plt.figure(figsize = (10,6))
sns.kdeplot(train_data['Fare'], label = "Skewness {:.2f}".format(train_data['Fare'].skew()))


# The Fare graph is very skewed and needs to be transformed. I'll use the log transform for that

# In[ ]:


train_data['Fare'] = train_data['Fare'].apply(lambda x: 0 if x == 0 else np.log(x))


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(train_data['Fare'])


# Now the fare column is log transformed 

# ### Embarked

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Survived', hue = 'Embarked', data=train_data)
plt.legend(bbox_to_anchor=[0.20,0,1,1])


# In[ ]:


plt.figure(figsize=(8,6))
g = sns.barplot(x = 'Embarked', y='Survived', data=train_data)


# This shows that the people from Cherbourg(C) had higher probability to survive than Queenstown (Q), Southampton (S).

# ## Imputation - Filling the Missing values

# ### Age

# In[ ]:


train_data['Age'].isnull().sum()


# The data has 177 Missing values. There are various methods of imputation but I'm imputing based on the highly correlated. 

# In[ ]:


train_data.corrwith(train_data['Age'])


# Fare has very less correlation with the Age. But Pclass, Sibsp and Parch has good -ve correlation. I'll take the Pclass for imputing the Age. As the age increases the people tend spend money on comfort and hence older the peole better the class

# ### Taking Pclass as the reference to impute the Missing Ages

# In[ ]:


plt.figure(figsize = (8,6))
sns.boxplot(x='Pclass', y="Age", data=train_data)


# In[ ]:


print(math.ceil(train_data[train_data['Pclass'] == 1]['Age'].mean()))
print(math.ceil(train_data[train_data['Pclass'] == 2]['Age'].mean()))
print(math.ceil(train_data[train_data['Pclass'] == 3]['Age'].mean()))


# I'm taking the Mean of the Age of particular Class for imputation

# In[ ]:


def age_impute(cols):
    age = cols[0]
    pclass = cols[1]
    if np.isnan(age) :
        if pclass == 1:
            return 39
        elif pclass == 2:
            return 30
        else :
            return 26
    else :
        return age
    
train_data['Age'] = train_data[['Age', 'Pclass']].apply(age_impute,axis=1)

# Converting all ages to their Upper bound
train_data['Age'] = train_data['Age'].apply(math.ceil)


# ### Cabin

# In[ ]:


train_data['Cabin'].isnull().sum()


# Very few people had Cabin. 687 people didn't have cabin so Let's fill the Null values of Cabin column with others

# In[ ]:


train_data['Cabin'] = train_data['Cabin'].fillna('Other')


# ## Feature Engineering

# ### Ticket

# I'm taking the first few words of the ticket as they might give me the clusters of people who sat together or somewhere near to each other. 

# In[ ]:


train_data['Ticket'] = train_data['Ticket'].apply(lambda x : x.split()[0])


# In[ ]:


train_data['Ticket'].value_counts()[:10]


# In[ ]:


def Ticket_categorize(tick):
    if tick == 'PC':
        return 'PC'
    elif tick == 'C.A.':
        return 'CA'
    elif tick == 'STON/O':
        return 'STON'
    else:
        return 'Other'


# In[ ]:


train_data['Ticket'] = train_data['Ticket'].apply(Ticket_categorize)


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = 'Ticket', y= 'Survived', data=train_data)


# In[ ]:


train_data.corrwith(train_data['Ticket'].map({'PC' : 4, 'CA': 3, 'STON' : 2, 'Other': 1}))


# As the fare is positively correlated to the tickets in above manner, we see that PC is the most expensive and people with that had more probablity for survival as compared to other. Then comes tickets with CA and then STON and least for Others 

# ### Cabin

# Adding a main_cabin column which is just the first alphabet of the Cabin

# In[ ]:


train_data['Cabin'] = train_data['Cabin'].dropna().apply(lambda x: x[0])
train_data['Cabin'].value_counts()


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x ='Cabin', y='Survived', data=train_data.dropna(),order='A B C D E F G T O'.split(' '))


# In[ ]:


train_data.corrwith(train_data['Cabin'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7,'O':8}))


# ### Title

# In[ ]:


train_data['Name'].head()


# Fetching the Titlte of each person 

# In[ ]:


train_data['Name'].apply(lambda x : x.split(',')[1].split('.')[0]).value_counts()


# In[ ]:


train_data['Name'] =  train_data['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())

## Keeping the Mr, Mrs , Miss, Master as it is, and replacing the others with Others

train_data['Name'] = train_data['Name'].replace(
    to_replace = ['Dr' , 'Rev', 'Mlle', 'Col', 'Major', 'Jonkheer', 'the Countess', 'Ms', 'Lady', 'Capt', 'Don', 'Mme' ], 
    value=  'Others')


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = train_data['Name'], y= train_data['Survived'])


# From the above graph we can interpret that the people with Mr. Title had very less survival rate whereas the Sir title had a lot more than other. Females i.e Mrs and Miss had great probability to survive than others and Master

# ### Adding the Family size column 

# In[ ]:


train_data['FamilySize'] = train_data['Parch'] + train_data['SibSp'] + 1


# In[ ]:


def cal_family(col):
    if col == 1:
        return 'Single'
    elif col > 1 and col <= 4 :
        return 'Small'
    else:
        return 'Large'


# In[ ]:


train_data['FamilySize']  = train_data['FamilySize'].apply(cal_family)


# In[ ]:


plt.figure(figsize = (8,6))
sns.barplot(x = 'FamilySize' , y = 'Survived', data = train_data)


# ### Sex into Male 1 and Female 0

# In[ ]:


train_data['Sex'] = train_data['Sex'].map({'male' :1, 'female':0})


# In[ ]:


train_data.head()


# ### Categorical Columns for Categorical Features

# In[ ]:


columns = [ 'Name', 'Ticket','Cabin', 'Embarked', 'FamilySize']


# In[ ]:


train_data = pd.concat([train_data, pd.get_dummies(train_data[columns], drop_first=True)],axis=1)


# ### Dropped the Cabin, Ticket, Name, Embarked, title,  columns

# In[ ]:


train_data = train_data.drop(columns,axis=1)


# ### Splitting the Features and Targets

# In[ ]:


train_data


# In[ ]:


train_data.isnull().sum()


# In[ ]:


X = train_data.drop('Survived',axis=1).astype(int)
y = train_data['Survived'].astype(int)


# ## Modeling

# ### Pipelines
# 

# In[ ]:


pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledSVC', Pipeline([('Scaler', StandardScaler()),('SVC', SVC())])))
pipelines.append(('ScaledDT', Pipeline([('Scaler', StandardScaler()),('DT', DecisionTreeClassifier())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))
pipelines.append(('ScaledGB', Pipeline([('Scaler', StandardScaler()),('GB', GradientBoostingClassifier())])))
pipelines.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostClassifier())])))
pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))
pipelines.append(('ScaledKN', Pipeline([('Scaler', StandardScaler()),('KN', KNeighborsClassifier())])))


# In[ ]:


column_values = ['Name', 'Score', 'Std Dev']
model_comp = pd.DataFrame(columns= column_values)


# In[ ]:


row = 0
for name, model in pipelines:
    kfold = KFold(n_splits=10)
    cv_score = cross_val_score(model,X, y, cv = kfold, scoring = 'accuracy', verbose = 2, n_jobs= -1)
    print('{} model -  accuracy mean {}  std {}'.format(name, cv_score.mean(),cv_score.std()))
    model_comp.loc[row, 'Name'] = name
    model_comp.loc[row, 'Score'] = cv_score.mean()
    model_comp.loc[row, 'Std Dev'] = cv_score.std()
    row += 1 


# In[ ]:


model_comp.sort_values(by='Score', ascending = False)


# So we'll take the first 6 models further for tuning them.

# ## Standardization

# In[ ]:


X


# In[ ]:


columns_to_scale = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
scaler = StandardScaler().fit(X[columns_to_scale])
X_scaled = scaler.transform(X[columns_to_scale])


# The categorical columns needn't be standarized because they are only in ones and zeros. Therefore we are standardizing all the columns but the categorical/dummies

# In[ ]:


columns_left = list(set(X.columns) - set(columns_to_scale)) ## Left out columns


# In[ ]:


X_scaled = np.column_stack((X_scaled, X[columns_left].values)) ## combining the standardized columns and categorical columns


# In[ ]:


X_scaled.shape


# ## GridSearchCV  - For the tuning our models

# In[ ]:


model_comp_after_gridsearch = pd.DataFrame(columns = ['Name', 'Best Score'])


# In[ ]:


models_param_grid  = [
                { 'C' : [0.5,0.8,1,1.5,2], 'max_iter' : [100,120,150,200] },
    
                { 'C' : [0.1,1,5,10,15],
              'gamma' : [0.1,0.03,0.05,0.08,0.5,0.8],
              'degree' : [2,3,5,8]},
                
                { 'learning_rate' : [0.01, 0.05, 0.1, 0.5, 0.8, 1],
                   'max_depth' : [2,3,4,5],
                    'n_estimators': [ 70, 90, 100,105, 110,115, 120 ]},
                
                {'learning_rate' : [0.01, 0.05, 0.1, 0.5, 0.8, 1],
                    'n_estimators': [ 70, 90, 100,105, 110,115, 120 ] }  ,    
                      
                {  'leaf_size' : [20,25,30,35,40 ],
                    'n_neighbors' : [3,5,7,9,11]},      
                      
                {'n_estimators' : [75,95,100,105,110,120],
              'max_features' :  [5,8,10,12,14],
              'min_samples_split' : [1,3,5,7],
              'min_samples_leaf' : [1,3,5,7]}
                      
                      
               ]
models = [ ('LogisticRegression', LogisticRegression()),('SVM',SVC()) , ('GradientBoosting',GradientBoostingClassifier()), ('AdaBoost', AdaBoostClassifier()), 
          ('KNeighbors' , KNeighborsClassifier()), ('RandomForest',RandomForestClassifier())]


# In[ ]:


for i in range(0,len(models)) :
    
    if i == 0:
        logistic_regression_gridsearch_model = GridSearchCV(models[i][1], param_grid = models_param_grid[i] , 
                                            cv = kfold, verbose = 2, scoring = 'accuracy', n_jobs= -1)
        logistic_regression_gridsearch_model.fit(X_scaled,y)
        model_comp_after_gridsearch.loc[i, 'Best Score'] = logistic_regression_gridsearch_model.best_score_
        
    elif i == 1:
        svm_gridsearch_model = GridSearchCV(models[i][1], param_grid = models_param_grid[i] , 
                                            cv = kfold, verbose = 2, scoring = 'accuracy', n_jobs= -1)
        svm_gridsearch_model.fit(X_scaled,y)
        model_comp_after_gridsearch.loc[i, 'Best Score'] = svm_gridsearch_model.best_score_
        
    elif i == 2 :
        gradientboosting_gridsearch_model = GridSearchCV(models[i][1], param_grid = models_param_grid[i] , 
                                            cv = kfold, verbose = 2, scoring = 'accuracy', n_jobs= -1)
        gradientboosting_gridsearch_model.fit(X_scaled , y)
        model_comp_after_gridsearch.loc[i, 'Best Score'] = gradientboosting_gridsearch_model.best_score_
    
    elif i ==3 :
        adaboost_gridsearch_model = GridSearchCV(models[i][1], param_grid = models_param_grid[i],
                                                cv = kfold, verbose = 2, scoring = 'accuracy', n_jobs = -1)
        adaboost_gridsearch_model.fit(X_scaled, y)
        model_comp_after_gridsearch.loc[i , 'Best Score'] = adaboost_gridsearch_model.best_score_
        
    elif i == 4:
        knn_gridsearch_model = GridSearchCV(models[i][1], param_grid = models_param_grid[i],
                                           cv = kfold, verbose = 2, scoring = 'accuracy' , n_jobs = -1)
        knn_gridsearch_model.fit(X_scaled,y)
        model_comp_after_gridsearch.loc[i , 'Best Score'] = knn_gridsearch_model.best_score_
    
    else:
        randomforest_gridsearch_model = GridSearchCV(models[i][1], param_grid = models_param_grid[i] , 
                                            cv = kfold, verbose = 2, scoring = 'accuracy', n_jobs= -1)
        randomforest_gridsearch_model.fit(X_scaled , y)
        model_comp_after_gridsearch.loc[i, 'Best Score'] = randomforest_gridsearch_model.best_score_
        
    model_comp_after_gridsearch.loc[i, 'Name'] = models[i][0]
    


# In[ ]:


model_comp_after_gridsearch.sort_values('Best Score', ascending = False)


# ## Learning Curve

# In[ ]:


def learning_curve_plot(model, title, X, y, cv, train_sizes = np.linspace(0.1,1,10)):
    
    train_sizes, test_scores, train_scores = learning_curve(model, X, y, cv = cv, train_sizes= train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores , axis =1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize = (8,6))
    plt.title(title)
    plt.xlabel('Training sizes')
    plt.ylabel('Scores')
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean + train_scores_std, train_scores_mean - train_scores_std, alpha = 0.2)
    plt.fill_between(train_sizes, test_scores_mean + test_scores_std, test_scores_mean - test_scores_std, alpha = 0.2)

    plt.plot(train_sizes, train_scores_mean, 'o-', label = 'training score' )
    plt.plot(train_sizes, test_scores_mean, 'o-', label = 'test score' )
    
    return plt

g = g = learning_curve_plot(logistic_regression_gridsearch_model.best_estimator_,"Logistic_regression learning curves",X_scaled,y,cv=kfold)
g = learning_curve_plot(svm_gridsearch_model.best_estimator_,"SVM learning curves",X_scaled,y,cv=kfold)
g = learning_curve_plot(gradientboosting_gridsearch_model.best_estimator_,"GradientBoosting learning curves",X_scaled,y,cv=kfold)
g = learning_curve_plot(adaboost_gridsearch_model.best_estimator_, "AdaBoost Learning Curve", X_scaled, y, cv = kfold)
g = learning_curve_plot(knn_gridsearch_model.best_estimator_ , "KNN Learning Curve", X_scaled, y, cv = kfold)
g = learning_curve_plot(randomforest_gridsearch_model.best_estimator_,"RandomForest learning curves",X_scaled,y,cv=kfold)


# The models performs good with the training dataset Not much of overfitting can be observed

# ## Building a Voting Classifier

# Since the all three models performs well, I am combining all and using VotingClassifier of sklearn

# In[ ]:


logistic_regression_model = logistic_regression_gridsearch_model.best_estimator_
svm_model = SVC(C = 1, degree = 2, gamma = 0.03, probability = True)
randomforest_model = randomforest_gridsearch_model.best_estimator_
gradientboosting_model = gradientboosting_gridsearch_model.best_estimator_
adaboost_model = adaboost_gridsearch_model.best_estimator_
knn_model = knn_gridsearch_model.best_estimator_


# In[ ]:


voting_model = VotingClassifier( [ 
    ('Logistic_Regression', logistic_regression_model), 
    ("Random Forest", randomforest_model) ,
    ("SVM",svm_model), 
    ('GradientBoosting', gradientboosting_model),
    ("AdaBoost", adaboost_model), 
    ("knn_model", knn_model)
                                ] ,
                                voting='soft', n_jobs = -1)
voting_model.fit(X_scaled, y)


# ### Prediction

# In[ ]:


def preprocess_data(test_data) :
    ## age imputation
    test_data = test_data.reset_index(drop=True)
    test_data['Age'] =  test_data[['Age','Pclass']].apply(age_impute,axis=1)
    
    ## Cabin
    test_data['Cabin'] = test_data['Cabin'].fillna('Others')
    test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])
    
    ## Name
    test_data['Name'] = test_data['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
    test_data['Name'] = test_data['Name'].replace(
        to_replace = ['Dr' , 'Rev', 'Mlle', 'Col', 'Major', 'Jonkheer', 'the Countess', 'Ms', 'Lady', 'Capt', 'Don', 'Mme' ],
        value=  'Others')
    
    
    ## Ticket
    test_data['Ticket'] = test_data['Ticket'].apply(lambda x:x.split()[0])
    test_data['Ticket'] = test_data['Ticket'].apply(Ticket_categorize)
    
    ## Fare
    test_data['Fare'] = test_data['Fare'].apply(lambda x: 0 if x == 0 else np.log(x))
    test_data['Fare'] = test_data['Fare'].fillna(np.mean(test_data['Fare']))
    
    ## Embarked
    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    
    ## sex
    test_data['Sex'] = test_data['Sex'].map({'male' :1, 'female':0})
    
    # familysize
    test_data['FamilySize'] = test_data['Parch'] + test_data['SibSp'] + 1
    test_data['FamilySize']  = test_data['FamilySize'].apply(cal_family)

    
    ## categorical 
    test_cols = list(test_data.columns)
    train_categorical_cols = [ 'Cabin_B', 'Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G',
                              'Cabin_O' ,'Cabin_T','Embarked_Q', 'Embarked_S' ,'Name_Miss','Name_Mr',
                              'Name_Mrs','Name_Others', 'Name_Sir','Ticket_Other', 'Ticket_PC' ,'Ticket_STON','FamilySize_Single',
                                'FamilySize_Small']
    
    columns = ['Cabin', 'Embarked', 'Name', 'Ticket', 'FamilySize' ]
    test_categorical_cols = pd.get_dummies(test_data[columns])
    
    
    left_categorical_cols = list(set(train_categorical_cols) - set(list(test_categorical_cols.columns)))
    
    left_df = pd.DataFrame(columns = left_categorical_cols, data = np.zeros((test_data.shape[0],len(left_categorical_cols))))
    test_data = pd.concat([test_data, test_categorical_cols, left_df], axis=1) ## combining the left out columns and test categorical cols
    test_data = test_data[test_cols + train_categorical_cols]  ## Rearranging the columns
    
    #dropping
    test_data = test_data.drop(columns,axis=1)
    
    test_data = test_data.dropna()
    test_scaled_data = standardize_data(test_data)
    
    predictions = voting_model.predict(test_scaled_data)
    
    return predictions


def standardize_data(test_data):
    columns_to_scale = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    scaler = StandardScaler()
    scaler.fit(test_data[columns_to_scale])
    X_scaled = scaler.transform(test_data[columns_to_scale])
    columns_left = list(set(test_data.columns) - set(columns_to_scale)) ## Left out columns
    X_scaled = np.column_stack((X_scaled, test_data[columns_left].values)) ## combining the standardized columns and categorical columns
    return X_scaled


# In[ ]:


predictions = preprocess_data(test_data.drop(['PassengerId'], axis=1))


# In[ ]:


submission = pd.DataFrame(data = { 'PassengerId' : test_data['PassengerId'], 'Prediction' : predictions})


# In[ ]:


submission


# In[ ]:


submission.to_csv('Submission.csv', index=False)


# In[ ]:




