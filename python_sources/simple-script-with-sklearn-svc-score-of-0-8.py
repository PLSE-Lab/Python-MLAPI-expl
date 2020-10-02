## simple script to predict survival using SVC model

# load packages
import os

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('bmh')
color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats

# load train and test data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print(train.shape, test.shape)

# drop identifier column
train_id = train['PassengerId']
test_id = test['PassengerId']
del train['PassengerId']
del test['PassengerId']
train.info()

# combine train and test features
n_train = train.shape[0]
n_test = test.shape[0]
y = train['Survived'].values
df = pd.concat((train, test)).reset_index(drop=True)
del df['Survived']

# location on-board was important, so extract deck and room from cabin number
from itertools import groupby

def split_text(s):
        for k, g in groupby(s, str.isalpha):
            yield ''.join(g)

cabin = df['Cabin'].tolist()
d = {i:list(split_text(str(e))) for i, e in enumerate (cabin)}

deck = ['None'] * len(cabin)
room = np.full(len(cabin), np.nan)
for i, (k,v) in enumerate (d.items()):
    if v[0] != 'nan':
        deck[i] = v[0]
        if len(v) > 1:
            if v[1].isnumeric():
                room[i] = int(v[1])

# some tickets have prefixes, not sure why, extract first substring
ticket_prefix = ['None'] * len(cabin)
for i in np.arange(len(cabin)):
    tmp = df.loc[i, 'Ticket'].split(' ')
    if len(set(tmp)) == 2:
        ticket_prefix[i] = tmp[0].split('/')[0].replace('.','')

# copy original features dataframe, before adding the ones created
df2 = df.copy()
df2['Deck'] = deck
df2['Room'] = pd.Series(room)
df2['TicketVar'] = ticket_prefix
df2['FamilySize'] = df2['SibSp'] + df2['Parch'] + 1 # size of family on-board (single = 1)
df2['FamilyName'] = [i.split(',')[0] for i in df2['Name']] # placeholder for potentially grouping by family

# replace missing values for deck
mask = df2['Pclass'] == 3
df2.loc[mask, 'Deck'] = df2.loc[mask, 'Deck'].fillna('F') # most on F, a few on G deck
mask = df2['Pclass'] == 2
df2.loc[mask, 'Deck'] = df2.loc[mask, 'Deck'].fillna(df2.loc[mask, 'Deck'].mode()[0]) # most on D to F deck
mask = df2['Pclass'] == 1
df2.loc[mask, 'Deck'] = df2.loc[mask, 'Deck'].fillna(df2.loc[mask, 'Deck'].mode()[0])

# replace missing values for age
mask = ((df2['FamilySize'] == 1) | ((df2['SibSp'] == 1) & (df2['Parch'] == 0))) # most likely adults
df2.loc[mask, 'Age'] = df2.loc[mask, 'Age'].fillna(df2.loc[mask, 'Age'].mean())
df2.loc[~mask, 'Age'] = df2.loc[~mask, 'Age'].fillna(df2.loc[~mask, 'Age'].median()) # more younger children likely

# bin values for age, children and young adults more likely to survive
bins = [0, 2, 5, 10, 18, 35, 65, np.inf]
names = ['<2', '2-5', '5-10', '10-18', '18-35', '35-65', '65+']
df2['AgeRange'] = pd.cut(df2['Age'], bins, labels=names)

# clean up and bin the values for fare, not sure if it matters much
df2['Fare'].fillna(df2['Fare'].median(), inplace=True)
df2['Fare'] = df2['Fare'].astype(int)

bins = [0, 5, 10, 15, 30, 50, np.inf]
names = ['<5', '5-10', '10-15', '15-30', '30-50', '50+']
df2['FareRange'] = pd.cut(df2['Fare'], bins, labels=names)

# add a variable to distinguish families, using name and family size
df2['FamilyName'] = df2['FamilyName'] + '_' + df2['FamilySize'].astype(str)

# drop original categorical features that we replaced by other ones
df2.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# fill any remaining missing values, can do better than zeros for room later
df2['Embarked'].fillna(df2['Embarked'].mode()[0], inplace=True)
df2['Deck'].fillna('None', inplace=True)
df2['Room'].fillna(0, inplace=True)

#label encoding for categorical variables 
ls =['Sex', 'Embarked', 'Deck', 'AgeRange', 'FareRange', 'TicketVar', 'FamilyName']

from sklearn.preprocessing import LabelEncoder
for f in ls:
    print(f)
    lbl = LabelEncoder()
    lbl.fit(list(df2[f].values))
    df2[f] = lbl.transform(list(df2[f].values))
    
# copy and create dummies for categorical variables
df3 = df2.copy()
df3 = pd.get_dummies(df3)
#print(df3.shape)

# split between train and test now that features are ready
train = df3[:n_train]
test = df3[n_train:] 

X_train = np.asarray(train); y_train = y
X_test = np.asarray(test)

# set the parameters by cross-validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe = Pipeline(steps=[('scaler', StandardScaler()), ('estimator', SVC())])
param_grid=dict(estimator__kernel = ['rbf', 'linear'],
                estimator__C = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4],
                estimator__gamma = [0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 0.125, 0.2])

search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_train, y_train)
print(search.best_params_)

prediction = search.predict(X_test)
print(prediction[1:10])