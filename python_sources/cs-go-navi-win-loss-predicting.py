#!/usr/bin/env python
# coding: utf-8

# **This is XGBoost model that tries to predict whether NaVi CS:GO will win or lose their next match.**

# I also wanted to say big thank you to [Mateus Dauernheimer Machado](https://www.kaggle.com/mateusdmachado), who provided the dataset and helped me improve my model.

# In[ ]:


# Importing the libraries
print('Importing libraries...')
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import datetime
print('Success!')
print('Natus Vincere CS:GO Win Predictor 0.02 by Ravehorn.')


# In[ ]:


# Parameters of the call
def params(argument):
    if argument == 'team_1':
        return 'Natus Vincere'
    elif argument == 'team_2':
        return 'fnatic'
    elif argument == '_map':
        return 'Mirage'
    elif argument == 'rank_1':
        return 867
    elif argument == 'rank_2':
        return 583
    raise NameError


# In[ ]:


# Importing the dataset
print('Importing data...')
dataset = pd.read_csv('../input/csgo-professional-matches/results.csv')
dataset['date'] = pd.to_datetime(dataset['date'])
_2019 = datetime.date(2019, 1, 1)
_2019 = pd.to_datetime(_2019)
dataset = dataset.loc[dataset['date'] >= _2019]
dataset_p1 = dataset.loc[dataset['team_1'] == params('team_1')]
dataset_p2 = dataset.loc[dataset['team_2'] == params('team_1')]
dataset_p3 = dataset.loc[(dataset['team_1'] == params('team_2')) & (dataset['team_2'] != params('team_1'))]
dataset_p4 = dataset.loc[(dataset['team_2'] == params('team_2')) & (dataset['team_1'] != params('team_1'))]
dataset = pd.concat([dataset_p1, dataset_p2, dataset_p3, dataset_p4])
X = dataset.iloc[:, [1, 2, 3, 14, 15]].values
y = dataset.iloc[:, 6].values
print('Success!')


# In[ ]:


# Start function
def choice():
    print('Starting to learn...')
    user_choice = '1'
    if user_choice == '1':
        learn()
    elif user_choice == '2':
        print('Programm finished.')
    else:
        print('\nYou probably pressed the wrong button, try again!')
        choice()


# In[ ]:


# Learning accuracy function, %
def accuracy():
    global X
    global y
    print('Evaluating accuracy...')
    # Encoding categorical data
    onehotencoder = OneHotEncoder()
    X_new = onehotencoder.fit_transform(X[:, 0:2]).toarray()
    X_new = X_new[:, 1:]
    X = X[:, [3, 4]]
    X = np.concatenate((X, X_new), axis=1)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # Fitting XGBoost
    classifier = XGBClassifier(n_estimators=500, learning_rate=0.00001,
                               max_depth=7, subsample=0.8, colsample_bytree=0.8, gamma=5)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    avg_acc = (cm[0][0] + cm[1][1]) / cm_sum * 100
    # Returning average accuracy of the algorithm
    return avg_acc


# In[ ]:


# Main predict function
def learn():
    global X
    global y
    team_1 = params('team_1')
    team_2 = params('team_2')
    rank_1 = params('rank_1')
    rank_2 = params('rank_2')
    _map = params('_map')
    X_test = np.array([[team_1, team_2, _map, rank_1, rank_2]])
    # Encoding categorical data
    print('Encoding data...')
    onehotencoder = OneHotEncoder()
    X_new_transformed = onehotencoder.fit_transform(X[:, 0:2]).toarray()
    X_new_transformed = X_new_transformed[:, 1:]
    X_transformed = X[:, [3, 4]]
    X_transformed = np.concatenate((X_transformed, X_new_transformed), axis=1)
    X_test_new_transformed = onehotencoder.transform(X_test[:, 0:2]).toarray()
    X_test_new_transformed = X_test_new_transformed[:, 1:]
    X_test_transformed = X_test[:, [3, 4]]
    X_test_transformed = np.concatenate((X_test_transformed, X_test_new_transformed), axis=1)
    print('Success!')
    print('Splitting the dataset...')
    # Splitting the dataset into the Training set and Test set
    X_train, X_test_old, y_train, y_test_old = train_test_split(X_transformed, y, test_size=1)
    print('Success!')
    print('Fitting XGBoost...')
    # Fitting XGBoost
    classifier = XGBClassifier(n_estimators=500, learning_rate=0.00001,
                               max_depth=7, subsample=0.8, colsample_bytree=0.8, gamma=5)
    classifier.fit(X_train, y_train)
    print('Success!')
    print('Predicting')
    # Predicting
    y_pred = classifier.predict(X_test_transformed)
    print('Success!')
    accuracy_final = accuracy()
    # Predicting the Test set results
    print(f'Accuracy of the algorithm: {accuracy_final}%')
    if y_pred == 2:
        print(f'Algorithm predicts that {team_1} will win {team_2}.')
    elif y_pred == 1:
        print(f'Algorithm predicts that {team_2} will win {team_1}.')
    else:
        print('An unexpected error has occured.')
    print('Programm finished.')


# In[ ]:


choice()


# **Thank you for trying out my model!**

# P.S. This is my first notebook and I'm new to machine learning. I know there may be a lot of bugs, issues, etc. And I would be very keen if somebody could help me, or criticise me, because I really want to get better. I'm always open to being criticised.
# In the middle of the development, I realised that it's pretty hard to predict the outcome of the match even using ML algorithms (those I know, of course).
# This is why I first calculated the function for evaluating overall model accuracy, and then I used it to predict the actual outcome of the match that could have possibly happened.
# Don't expect it to be higher than 70%. If it is higher - it may be biased and/or overfitted (as far as I know for esports).

# P.S. I also decided not to use the ELO algorithm, since it has certain drawbacks and is already implemented. From my understanding, ML, DL, ELO or MMR for example could underperform in 5v5 competitive games, such as CS:GO, Dota 2, etc, because of many factors. But I decided to try ML, because that is what brings me joy.
# I'm very happy that I tried and did this research.

# Again, big thanks to the database scraper, you gave me a wonderful opportunity to explore and learn!
