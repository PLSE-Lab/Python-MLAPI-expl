#!/usr/bin/env python
# coding: utf-8

# # Importing Data

# Inspo taken from Ayoub - learning the basics :) 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_filename = os.path.join("../input/titanic/train.csv")
test_filename = os.path.join("../input/titanic/test.csv")
train = pd.read_csv(train_filename, encoding='latin1')
test = pd.read_csv(test_filename, encoding='latin1')

train.head()


# # Data Exploration

# In[ ]:


print(train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean())
plt.figure(figsize=(12, 8))
sns.boxplot(x='Pclass', y='Age', hue='Survived', data=train)


# In[ ]:


survived = train.loc[(train['Survived']==1)]
total_passengers = len(train)
total_survivors = len(survived)
number_female_survivors = len(survived.loc[(survived['Sex']=='female')])
number_male_survivors = total_survivors - number_female_survivors
dist_male_female = [number_female_survivors, number_male_survivors]
print('Total Passengers: {} \n'.format(total_passengers))
print('Total Number of Survivors: {} \n'.format(total_survivors))
print('Number of Female Survivors: {} \n'.format(number_female_survivors))
print('Number of Male Survivors: {} \n'.format(number_male_survivors))
sns.countplot(x='Survived', data=train, hue='Sex')


# In[ ]:


plt.pie(dist_male_female, labels=['female', 'male'])


# In[ ]:


male_survivors = survived.loc[(survived['Sex']=='male')]
male_survivors_ages = male_survivors['Age'].dropna()
sns.distplot(male_survivors_ages)


# In[ ]:


female_survivors = survived.loc[(survived['Sex']=='female')]
female_survivors_ages = female_survivors['Age'].dropna()
sns.distplot(female_survivors_ages)


# In[ ]:


train[["Sex", "Survived"]].groupby('Sex', as_index=False).mean()


# In[ ]:


g = sns.FacetGrid(train, col='Survived', row='Pclass')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


sns.countplot(x='Embarked', data=train, hue='Survived')


# In[ ]:


sns.countplot(x='Parch', data=train, hue='Survived')


# In[ ]:


sns.countplot(x='SibSp', data=train, hue='Survived')


# In[ ]:


sns.catplot(x="SibSp",y="Survived",data=train,kind="bar")


# # Data Cleaning

# In[ ]:


train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)


# In[ ]:


from collections import Counter

def detect_outliers(df,n,features):
    outlier_indices = []
    for col in features:
        # first quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # third quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # interquartile range
        IQR = Q3 - Q1
        
        # Outlier Step
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)    
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

Outliers_to_drop = detect_outliers(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])


# In[ ]:


train.loc[Outliers_to_drop]


# In[ ]:


train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
train_len = len(train)


# In[ ]:


combined = pd.concat([train, test], ignore_index=True)


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


np.sum(train.isnull())


# In[ ]:


np.sum(test.isnull())


# ## Process embarked (fill 2 NaN values)

# In[ ]:


embarked_mode = train['Embarked'].mode()[0]


# In[ ]:


combined['Embarked'] = combined['Embarked'].fillna(embarked_mode)


# ## Process Cabin Column

# In[ ]:


combined['Cabin'] = combined['Cabin'].apply(str).apply(lambda x: x[0])


# In[ ]:


combined['Cabin'].unique()


# In[ ]:


combined['Cabin'].fillna('U', inplace=True)


# In[ ]:


sns.countplot(combined['Cabin'])


# In[ ]:


np.where(np.isnan(test['Fare']))


# In[ ]:


test.iloc[152]


# In[ ]:


sns.heatmap(train[['Fare', 'Age', 'Sex', 'SibSp', 'Parch', 'Pclass' ]].corr(), annot=True, cmap='coolwarm')


# In[ ]:


train[(train['Pclass'] == 3)]['Fare'].median()
combined.loc[(combined['Name'] == 'Storey, Mr. Thomas')]


# In[ ]:


combined['Fare'][1033] = train[(train['Pclass']==3)]['Fare'].median()


# In[ ]:


combined['Fare'] = combined['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
np.sum(combined.isnull())


# # Feature Engineering

# Process name

# In[ ]:


train['Name'].head()


# Get title of each passenger

# In[ ]:


combined['Title'] = combined['Name'].apply(lambda x: (x.split(', ')[1].split('.')[0]))
combined.drop('Name', axis=1, inplace=True)
combined['Title'].unique()


# In[ ]:


combined['Title'].value_counts()


# In[ ]:


titles_map = {
 'Capt' : 'Rare',
 'Col' : 'Rare',
 'Don': 'Rare',
 'Dona': 'Rare',
 'Dr' : 'Rare',
 'Jonkheer' :'Rare' ,
 'Lady': 'Rare',
 'Major': 'Rare',
 'Master': 'Master',
 'Miss' : 'Miss',
 'Mlle' : 'Rare',
 'Mme': 'Rare',
 'Mr': 'Mr',
 'Mrs': 'Mrs',
 'Ms': 'Rare',
 'Rev': 'Rare',
 'Sir': 'Rare',
 'the Countess': 'Rare'    
}


# In[ ]:


combined['Title'] = combined['Title'].map(titles_map)


# In[ ]:


combined[['Title', 'Survived']].groupby('Title', as_index=False).mean()


# In[ ]:


combined.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder



#embarked_encoder = LabelEncoder().fit(combined['Embarked'])
#combined['Embarked'] = embarked_encoder.transform(combined['Embarked'])
combined = pd.get_dummies(combined, columns=['Embarked'], prefix='Em')

combined = pd.get_dummies(combined, columns=['Cabin'], prefix='Cb')



combined = pd.get_dummies(combined, columns=['Title'], prefix='Title')


# In[ ]:


sex_encoder = LabelEncoder().fit(combined['Sex'])
combined['Sex'] = sex_encoder.transform(combined['Sex'])


# process age

# In[ ]:


sns.heatmap(train[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']].corr(), annot=True, cmap='coolwarm')


# In[ ]:


def impute_age(row):
    pclass = row['Pclass']
    parch = row['Parch']
    sibsp = row['SibSp']
    age = row['Age']
    if pd.isnull(age):
        age_median = train['Age'].median()
        similar_age = train[(train['Pclass'] == pclass) & 
                            (train['Parch'] == parch) & 
                            (train['SibSp'] == sibsp)]['Age'].median()
        if (similar_age > 0): return similar_age
        else: return age_median
    else: return age    
    


# In[ ]:


combined['Age'] = combined.apply(impute_age, axis=1)


# In[ ]:


np.sum((combined.drop('Survived', axis=1).isnull()))


# process parch and sibsp

# In[ ]:


combined['Family_size'] = combined.apply(lambda row: 1 + (row['Parch'] + row['SibSp']), axis=1)
combined['Alone'] = combined.apply(lambda row: 1 if (row['Parch'] + row['SibSp']) == 0 else 0, axis=1)


# In[ ]:


sns.countplot(x='Family_size', data=combined, hue='Survived')


# In[ ]:


combined['Small_family'] = combined.apply(lambda row: 1 if 2 <= (row['Family_size']) <= 4 else 0, axis=1)
combined['Large_family'] = combined.apply(lambda row: 1 if (row['Family_size']) > 4 else 0, axis=1)


# In[ ]:


combined.head()


# combine Pclass & age

# In[ ]:


combined['Age*Pclass'] = combined['Age'] * combined['Pclass']


# Process ticket

# In[ ]:


import re


# In[ ]:


combined['Ticket'] = combined['Ticket'].apply(lambda x: 'X' if x.isdigit() else x)


# In[ ]:


combined['Ticket'] = combined['Ticket'].apply(lambda x: re.sub("[\d\.]", "", x).split('/')[0].strip() if not x.isdigit() else x)


# In[ ]:


ticket_encoder = LabelEncoder().fit(combined['Ticket'])
combined['Ticket'] = ticket_encoder.transform(combined['Ticket'])


# # Model Machine Learning

# In[ ]:


train = combined[:train_len]
test = combined[train_len:]


# In[ ]:


test.drop('Survived', axis=1, inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


np.sum(test.isnull())


# In[ ]:


train.head()


# In[ ]:


train['Survived'] = train['Survived'].astype(int)


# generating random samples of the data to test the models for accuracy

# In[ ]:


from sklearn.preprocessing import StandardScaler

X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# Checking the accuracy of each model using the random samples generated in the previous step

# In[ ]:


random_state = 42

model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'SVC', 
              'RandomForestClassifier', 'XGBClassifier', 'ExtraTreesClassifier'
              , 'GradientBoostingClassifier','AdaBoostClassifier','GaussianNB']

models = [ ('LogisticRegression',LogisticRegression(random_state=random_state)),
          ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
          ('SVC', SVC(random_state=random_state)),
          ('RandomForestClassifier',RandomForestClassifier(random_state=42)),
          ('XGBClassifier',XGBClassifier(random_state=random_state)),
          ('ExtraTreesClassifier',ExtraTreesClassifier(random_state=random_state)),
          ('GradientBoostingClassifier',GradientBoostingClassifier(random_state=random_state)),
          ('AdaBoostClassifier',AdaBoostClassifier(random_state=random_state)),
          ('GaussianNB',GaussianNB())
         ]

model_accuracy = []

for k, model in models:
    print(k, ':')
    model.fit(X, y)
    accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
    model_accuracy.append(accuracy)
    print(accuracy)
    print('\n')


# Putting the model names next to the accuracy of each model for the sample data sets.

# In[ ]:


pd.concat([pd.Series(model_names), pd.Series(model_accuracy)], axis=1).sort_values(by=1, ascending=False)


# ## Hyperparameter tuning

# Using a grid to search for the best estimator

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


best_models = []

xgboot_param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [.4, .45, .5, .55, .6], 
    'colsample_bytree': [.6, .7, .8, .9, 1]
}

ada_param_grid = {
 'n_estimators':[100,200,300],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 'algorithm' : ['SAMME', 'SAMME.R']
 }

gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,200,300],
              "criterion": ["gini"]}

rf_param_grid  = { 
    'n_estimators': [100,200,300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

log_param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

svv_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

models = [ 
    ('AdaBoostClassifier',AdaBoostClassifier(), ada_param_grid),
          ('XGBClassifier',XGBClassifier(), xgboot_param_grid),
          ('GradientBoostingClassifier',GradientBoostingClassifier(), gb_param_grid),
        ('RandomForestClassifier',RandomForestClassifier(), rf_param_grid),
          ('ExtraTreesClassifier',ExtraTreesClassifier(), ex_param_grid),
    ('SVC',SVC(probability=True), svv_param_grid),
    ('LogisticRegression',LogisticRegression(), log_param_grid)
         ]

for name, model, param in models: 
    print(name, ':')
    grid_search = GridSearchCV(model, 
                              scoring = 'accuracy',
                              param_grid=param,
                              cv=5,
                              verbose=2,
                              n_jobs=-1) 
    grid_search.fit(X, y)
    print(grid_search.best_score_, '\n')
    best_models.append(grid_search.best_estimator_)


# In[ ]:


test


# In[ ]:


from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None: 
        plt.ylim(*ylim)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, 
                                                            X, y, 
                                                            cv=cv, 
                                                            n_jobs=n_jobs, 
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, 
                     color='r')
    plt.fill_between(train_sizes,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1,
                    color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross Validation Score')
    
    plt.legend(loc='best')
    return plt

for model in best_models: 
    plot_learning_curve(model, model.__class__.__name__ + 'RF Learning Curves', X, y, cv=10)
    
    
    


# In[ ]:


def plot_feature_importances(clf, X_train, 
                            y_train=None, top_n=10, 
                            figsize=(8,8), print_table=False, 
                            title='Feature Importances'):
    __name__ = 'plot_feature_importances'
        
    from xgboost.core     import XGBoostError
    from lightgbm.sklearn import LightGBMError
    
    try: 
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())
            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError('{} Does not have feature_importances_ attribute'.format(clf.__class__.__name__))
    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())
    
    feat_imp = pd.DataFrame({'importance': clf.feature_importances_})
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table: 
        from IPython.display import display
        print('Top {} features according to importance'.format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp

        
    


# In[ ]:


for model in best_models:
    try:
        _ = plot_feature_importances(model, X_train, y_train, top_n=X.shape[1], title=model.__class__.__name__)

    except AttributeError as e:
        print(e)
        
        


# In[ ]:


pred = []
for model in best_models: 
    pred.append(pd.Series(model.predict(test), name=model.__class__.__name__))
    
    


# In[ ]:


pred = pd.DataFrame(pred).T
pred


# In[ ]:


pred.sum()


# In[ ]:


g = sns.heatmap(pred.corr(), annot=True, cmap='coolwarm')


# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


best_models


# In[ ]:


votingC = VotingClassifier(estimators = [
    ('ada', best_models[0]),
    ('rf', best_models[3]),
    ('ext', best_models[4]), 
    ('scv', best_models[5]),
    ('log', best_models[6]),
], voting='soft', n_jobs=5)

votingC.fit(X, y)


# In[ ]:


test_Survived = pd.Series(votingC.predict(test), name='Survived')

results = pd.concat([pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId'],test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)

results


# In[ ]:


ext_best = best_models[4]
ext_best.fit(X, y)


# In[ ]:


test_Survived = pd.Series(ext_best.predict(test), name="Survived")

results = pd.concat([pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId'],test_Survived],axis=1)

results.to_csv("prediction.csv",index=False)

