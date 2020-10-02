# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization 
from sklearn.model_selection import train_test_split # create train and test datasets
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn import metrics # evaluate performance of model 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Read in data and output first 10 entries 
df = pd.read_csv('../input/master.csv')
df.head()


# shape of data 
df.shape

# --------- Predicting using Linear Regression -------------

# fit model and perform k-fold CV to compute error
def error (clf, X, y, ntrials=100, test_size=.2): 
    
    test_error = 0
    train_error = 0
    
    i = 0
    while i < ntrials:
        # split data into train and test tests 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = i)

        # if X has one feature, reshape into 2D array
        X_shape = X.shape
        if (len(X_shape) == 1): 
            X_train = X_train.values.reshape(-1, 1)
            X_test = X_test.values.reshape(-1, 1)
        
        # fit model 
        clf.fit(X_train, y_train)

        # make predictions 
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # compute error for train and test (MSE)
        train_error += metrics.mean_squared_error(y_train, y_train_pred)
        test_error += metrics.mean_squared_error(y_test, y_test_pred)
        
        i += 1

    # compute average error over ntrials
    train_error = train_error/ntrials
    test_error = test_error/ntrials 
    return train_error, test_error 

# --------------- clean features -------------

# label encode age
labelencoder = LabelEncoder()
pd.options.mode.chained_assignment = None
df['age'].copy = df['age']
original_age = df['age'].copy # store a copy for graph labels 
df['age'] = labelencoder.fit_transform(df['age'])

# one hot encode sex
df['sex'] = pd.Categorical(df['sex'])
dfDummies = pd.get_dummies(df['sex'], prefix = 'sex')
df = pd.concat([df, dfDummies], axis=1)

# label encode generation 
df['generation'].copy = df['generation']
original_gen = df['generation'].copy
df['generation'] = labelencoder.fit_transform(df['generation'])

# one hot encode country 
df['country'] = pd.Categorical(df['country'])
dfDummies = pd.get_dummies(df['country'], prefix = 'country')
df = pd.concat([df, dfDummies], axis=1)

# create four subsets of df based on geographic region: West, Far East, Middle, Hispanic 


# ------------ test model ---------------

# create X with cleaned features 
X = df[['population', 'sex_male', 'age', 'generation', 'gdp_per_capita ($)', 'year']]
# predict suicides per 100k people 
y = df['suicides/100k pop']

# compute error using linear regression 
reg = LinearRegression()    
train_error, test_error = error(reg, X, y)
print ("Error using OLS")
print ("Train error: ", train_error)
print ("Test error: ", test_error)

#
reg = Ridge(alpha=1)
train_error, test_error = error(reg, X, y, ntrials=100, test_size=.2)
print ("Error using ridge regression")
print ("Train error: ", train_error)
print ("Test error: ", test_error)

# compute error using neural network 
reg = MLPRegressor(solver='lbfgs')
train_error, test_error = error(reg, X, y, ntrials = 5)
print ("Train error: ", train_error)
print ("Test error: ", test_error)

# ----------------------- Visualization -------------------------

plt.figure(figsize=(16,8))
corr_table = sns.heatmap(df.corr(), annot=True)

# graph number of suicides with sex and age group 
ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(x = original_age.sort_values(),y = 'suicides_no',hue='sex',data=df,palette='rainbow')


list(df.columns.values)
# graph generation and suicide number
ax = plt.subplots(figsize=(16,10))
ax = sns.barplot(x = original_gen.sort_values(),y = 'suicides_no',data=df)

# sex and suicide number
ax = plt.subplots(figsize=(16,10))
ax = sns.barplot(x = df['sex'].sort_values(),y = 'suicides_no',data=df)

# age and suicides 
ax = plt.subplots(figsize=(16,10))
ax = sns.scatterplot(x = df['year'],y = 'suicides_no',data=df)

# histogram of year
ax = sns.distplot(df['year'])

# ----------------- interpreting the data -------------------

# examine relationship between distinct geographic regions and suicide rate

# create four subsets based on geographic region: Scandanavia, Far east, Latin America, West, 
list(df)
west = df.loc[(df['country'] == 'Canada') | (df['country'] == 'United States') | 
                (df['country'] == 'Ireland') | (df['country'] == 'France') | (df['country'] == 'Australia')
                | (df['country'] == 'United Kingdom')]
far_east = df.loc[(df['country'] == 'Japan') | (df['country'] == 'Philippines') | 
                    (df['country'] == 'Republic of Korea')]
scand = df.loc[(df['country'] == 'Denmark') | (df['country'] == 'Finland') | (df['country'] == 'Sweden')
                | (df['country'] == 'Norway')]
latin_america = df.loc[(df['country'] == 'Nicaragua') | (df['country'] == 'Panama') |
                        (df['country'] == 'Paraguay') | (df['country'] == 'Brazil') | 
                        (df['country'] == 'Argentina') | (df['country'] == 'Mexico')]
















    
    









