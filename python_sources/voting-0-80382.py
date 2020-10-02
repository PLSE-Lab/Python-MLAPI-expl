#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import other needed packages and functions
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import lightgbm
from sklearn.model_selection import cross_val_score
import itertools


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read in csv files and make dfs
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

# make copy of test df for submission
submission = test_df.copy()

# combine train and test dfs into 1 df of all data
all_df = pd.concat([train_df, test_df], sort=False)


# In[ ]:


# Define function to inspect data frames. Prints first few lines, determines size/shape of data frame,
# shows descriptive statistics, shows data types, shows missing or incomplete data, check for duplicate data.

def inspect_df(df):
    print('Header:')
    print('{}'.format(df.head()))
    print()
    print('Shape: {}'.format(df.shape))
    print()
    print('Statistics:')
    print('{}'.format(df.describe()))
    print()
    print('Info:')
    print('{}'.format(df.info()))
    
# use inspect_df on all_df

inspect_df(all_df)


# In[ ]:


# look at proportions of passengers by Pclass

all_df.Pclass.value_counts(normalize=True, sort=False)


# In[ ]:


# look at proportions of passengers by Sex

all_df.Sex.value_counts(normalize=True)


# In[ ]:


# inspect null values for Embarked

all_df[all_df.Embarked.isnull()]


# Looked up Mrs. Stone and Miss Icard online, they boarded in Southampton.

# In[ ]:


# fill missing values for Embarked with information found online

all_df.loc[[61, 829], ['Embarked']] = 'S'


# In[ ]:


# plot histogram of Ages

plt.hist(data = all_df, x = 'Age', bins = 40);


# In[ ]:


# plot histogram of Fare

plt.hist(data = all_df, x = 'Fare', bins = 40);


# In[ ]:


# plot histogram of siblings and spouses

plt.hist(data = all_df, x = 'SibSp', bins = 8);


# In[ ]:


# plot histogram of parents and children

plt.hist(data = all_df, x = 'Parch', bins = 9);


# In[ ]:


# inspect columns and missing values again

all_df.info()


# In[ ]:


# find passenger with missing fare data

all_df[all_df.Fare.isnull()]


# In[ ]:


# get average fare of each Pclass

all_df.Fare.groupby(all_df.Pclass).mean()


# In[ ]:


#fill nan fare value with rounded mean for class 3

all_df.loc[152, ['Fare']] = 13


# In[ ]:


# fill missing values for ages with the mean age value for each passengers Pclass and Sex

all_df.Age = all_df.Age.groupby([all_df.Pclass, all_df.Sex]).transform(lambda x: x.fillna(x.mean()))


# In[ ]:


# Extract titles from names and make new Title column, then get survival rate of each title

all_df['Title'] = all_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
all_df.Survived.groupby(all_df.Title).mean()


# In[ ]:


# get value counts for title occurences

all_df.Title.value_counts()


# In[ ]:


# Replace uncommon titles with more common values and view new occurences

all_df.Title = all_df.Title.replace(['Capt', 'Col', 'Dr', 'Major', 'Rev', 'Don', 'Sir', 'Jonkheer'], 'Mr')
all_df.Title = all_df.Title.replace(['Ms', 'Mlle'], 'Miss')
all_df.Title = all_df.Title.replace(['Mme', 'Lady', 'Countess', 'Dona'], 'Mrs')
all_df.Title.value_counts()


# In[ ]:


# combine sibsp and parch to one family column
all_df['Fam'] = all_df.SibSp + all_df.Parch

# make ticket frequency column for number of occurences of ticket number
all_df['Ticket_Frequency'] = all_df.groupby('Ticket')['Ticket'].transform('count')
    
# make column for solo vs travel with family
all_df.loc[all_df['Fam'] == 0, 'Solo'] = 1
all_df.loc[all_df['Ticket_Frequency'] == 1, 'Solo'] = 1
all_df.Solo = all_df.Solo.fillna(0)
       
# bin fare column to 9 quantiles and encode as ordinal
all_df['Fare'] = pd.qcut(all_df.Fare, q=9, labels=np.arange(1,10))
    
# bin age column to 10 quantiles and encode as ordinal
all_df['Age'] = pd.qcut(all_df.Age, q=10, labels=np.arange(1,11))
    
# one-hot encode sex column and capitalize sex columns for consistency
all_df = pd.concat([all_df, pd.get_dummies(all_df.Sex)], axis=1)
all_df.rename(columns={'male':'Male', 'female':'Female'}, inplace=True)
    
# one-hot encode embarked column
all_df = pd.concat([all_df, pd.get_dummies(all_df.Embarked, prefix='Embarked')], axis=1)
    
# one-hot encode title column
all_df = pd.concat([all_df, pd.get_dummies(all_df.Title)], axis=1)
    
# drop unwanted columns (name, sex, cabin, embarked and title have been replaced with one hot encoding, ticket replaced with ticket frequency,
# cabin has too many missing values, sibsp and parch replaced with fam and solo columns)
all_df = all_df.drop(columns=['Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Embarked', 'Title'])
    
# inspect columns and number of values for resulting df
all_df.info()


# In[ ]:


# inspect survival rates for number of family members onboard

all_df.Survived.groupby(all_df.Fam).mean()


# In[ ]:


# bin fam column values and get value counts

all_df.Fam = pd.cut(all_df.Fam, bins=[0, 1, 4, 7, 11], include_lowest=True, right=False, labels=[1, 2, 3, 4])
all_df.Fam.value_counts()


# In[ ]:


# inspect survival rate for number of people traveling in group

all_df.Survived.groupby(all_df.Ticket_Frequency).mean()


# In[ ]:


# bin ticket frequency column values and get value counts

all_df.Ticket_Frequency = pd.cut(all_df.Ticket_Frequency, bins=[0, 2, 5, 9, 12], right=False, labels=[1, 2, 3, 4])
all_df.Ticket_Frequency.value_counts()


# In[ ]:


# make list of all columns and view it

cols = list(all_df)
cols


# In[ ]:


# use min max scaler to scale all feature columns to range 0-1

scaler = MinMaxScaler()
all_df[cols] = scaler.fit_transform(all_df[cols])


# In[ ]:


# make array of survived labels (training), drop survived column from all_df, split all_df into features (training) and test_df, and make array of feature column names

labels = all_df.loc[:890, 'Survived']
all_df = all_df.drop(columns = 'Survived')
features = all_df.iloc[:891]
test_df = all_df.iloc[891:]
feat_names = features.columns.values


# In[ ]:


# use SelectKBest to narrow down to top features and use result to transform train and test features dfs

k = SelectKBest(k=11)
k.fit(features, labels)
k_scores = (k.scores_)
features = k.transform(features)
test_df = k.transform(test_df)


# In[ ]:


# make df to show scores for all features and print

feat_scores = pd.DataFrame()
feat_scores['Feature'] = feat_names
feat_scores['Score'] = k_scores
feat_scores


# In[ ]:


# split training data into train and test subsets for validation

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=3)


# In[ ]:


# setup base classifiers

gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
et = ExtraTreesClassifier()
ab = AdaBoostClassifier()
dt = DecisionTreeClassifier()
lr = LogisticRegression(solver='liblinear')
kn = KNeighborsClassifier()
svc = SVC(gamma='auto', probability=True)
gnb = GaussianNB()


# In[ ]:


# get train base classifiers and get initial validation scores

gb.fit(features_train, labels_train)
print('GB Score:', gb.score(features_test, labels_test))
rf.fit(features_train, labels_train)
print('RF Score:', rf.score(features_test, labels_test))
et.fit(features_train, labels_train)
print('ET Score:', et.score(features_test, labels_test))
ab.fit(features_train, labels_train)
print('AB Score:', ab.score(features_test, labels_test))
dt.fit(features_train, labels_train)
print('DT Score:', dt.score(features_test, labels_test))
lr.fit(features_train, labels_train)
print('LR Score:', lr.score(features_test, labels_test))
kn.fit(features_train, labels_train)
print('KN Score:', kn.score(features_test, labels_test))
svc.fit(features_train, labels_train)
print('SVC Score:', svc.score(features_test, labels_test))
gnb.fit(features_train, labels_train)
print('GNB Score:', gnb.score(features_test, labels_test))


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = gb
params = {'n_estimators': (10, 25, 50, 100), 'learning_rate': (0.01, 0.1, 0.5, 1, 5, 10)}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
gb = clf.best_estimator_


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = rf
params = {'n_estimators': (10, 25, 50, 100), 'min_samples_split': (2, 3, 4, 5, 10), 'min_samples_leaf': (1, 2, 3, 4, 5)}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
rf = clf.best_estimator_


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = et
params = {'n_estimators': (10, 25, 50, 100), 'min_samples_split': (2, 3, 4, 5, 10), 'min_samples_leaf': (1, 2, 3, 4, 5)}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
et = clf.best_estimator_


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = ab
params = {'n_estimators': (10, 25, 50, 100), 'learning_rate': (0.01, 0.1, 0.5, 1, 5, 10)}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
ab = clf.best_estimator_


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = dt
params = {'min_samples_split': (2, 3, 4, 5, 10), 'min_samples_leaf': (1, 2, 3, 4, 5)}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
dt = clf.best_estimator_


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = lr
params = {'penalty': ('l1', 'l2'), 'C': (0.01, 0.1, 0.5, 1, 5, 10), 'max_iter': (100, 500)}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
lr = clf.best_estimator_


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = kn
params = {'n_neighbors': (2, 3, 4, 5, 10, 20)}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
kn = clf.best_estimator_


# In[ ]:


# use gridsearch to tune base classifier hyperparameters

alg = svc
params = {'C': (0.01, 0.1, 0.5, 1, 5, 10), 'kernel': ('linear', 'poly', 'rbf', 'sigmoid')}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features_train, labels_train)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
svc = clf.best_estimator_


# In[ ]:


# retrain base classifiers and get validation scores

gb.fit(features_train, labels_train)
print('GB Score:', gb.score(features_test, labels_test))
rf.fit(features_train, labels_train)
print('RF Score:', rf.score(features_test, labels_test))
et.fit(features_train, labels_train)
print('ET Score:', et.score(features_test, labels_test))
ab.fit(features_train, labels_train)
print('AB Score:', ab.score(features_test, labels_test))
dt.fit(features_train, labels_train)
print('DT Score:', dt.score(features_test, labels_test))
lr.fit(features_train, labels_train)
print('LR Score:', lr.score(features_test, labels_test))
kn.fit(features_train, labels_train)
print('KN Score:', kn.score(features_test, labels_test))
svc.fit(features_train, labels_train)
print('SVC Score:', svc.score(features_test, labels_test))
gnb.fit(features_train, labels_train)
print('GNB Score:', gnb.score(features_test, labels_test))


# In[ ]:


# setup voting classifier and get initial cross val score

vote = VotingClassifier(estimators=[('gb',gb), ('rf',rf), ('et',et), ('ab',ab), ('dt',dt), ('lr',lr), ('kn',kn), ('svc',svc), ('gnb',gnb)], voting='soft')
vote.fit(features, labels)
cross_val_score(vote, features, labels, cv=5, scoring='accuracy').mean()


# In[ ]:


# retrain voting classifier and get validation score

vote.fit(features_train, labels_train)
print('Voting Score:', vote.score(features_test, labels_test))


# In[ ]:


# use itertools combinations to make list of tuples of all possible different combinations of classifiers

clfs = [('gb',gb), ('rf',rf), ('et',et), ('ab',ab), ('dt',dt), ('lr',lr), ('kn',kn), ('svc',svc), ('gnb',gnb)]
combs = []

for i in range(2, len(clfs)+1):
    comb = [list(x) for x in itertools.combinations(clfs, i)]
    combs.extend(comb)


# In[ ]:


# Tune voting classifier to use best combination of base classifiers

alg = vote
params = {'estimators': combs}
clf = GridSearchCV(alg, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
clf.fit(features, labels)
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)
vote = clf.best_estimator_


# In[ ]:


# retrain voting classifier and get validation score

vote.fit(features_train, labels_train)
print('Voting Score:', vote.score(features_test, labels_test))


# In[ ]:


# retrain voting classifier with full train set, use to make probability predictions and make df of probs, set threshold for probablities and use to convert probs to predictions

vote.fit(features, labels)
pred_prob = pd.DataFrame(vote.predict_proba(test_df))
threshold = 0.55
y_pred = pred_prob.applymap(lambda x: 1 if x>threshold else 0)


# In[ ]:


# add predictions submission df as survived column, drop all columns but passenger ID and survived, write submission to csv without index to generate submission file

submission['Survived'] = y_pred[1].astype(int)
submission = submission[['PassengerId', 'Survived']]
submission.to_csv('submission.csv', index=False)

