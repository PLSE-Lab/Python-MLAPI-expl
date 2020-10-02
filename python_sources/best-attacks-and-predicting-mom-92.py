#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/FIFA 2018 Statistics.csv')


# **Data Exploration**

# In[ ]:


data.head()


# In[ ]:


len(data)


# In[ ]:


data.isna().sum()


# NA values appear for 1st Goal, Own goals and Own goal Time (for the teams which didn't score)

# Goals scored

# In[ ]:


sns.set(rc={'figure.figsize':(8,6)})
sns.distplot(data['Goal Scored'])


# In[ ]:


data['Goal Scored'].value_counts(normalize=True)


# In[ ]:


goals_scored = 0
for i in range(0,3):
    goals_scored += data['Goal Scored'].value_counts(normalize=True)[i]
print("Teams scored less than 3 goals in %s%% of the games "%(100*round(goals_scored,4)))


# Teams scored between 0 and 2 goals in 88% of the games

# Best attacks

# In[ ]:


data.head()


# In[ ]:


attacks = data[['Team', 'Goal Scored', 'Round']]


# In[ ]:


all_attacks = attacks.groupby('Team').sum()
all_attacks = all_attacks.reset_index()
all_attacks = all_attacks.sort_values('Goal Scored', ascending=False)


# In[ ]:


g = sns.catplot(x='Team', y='Goal Scored', data=all_attacks, kind="bar", aspect=2)
g.set_xticklabels(rotation=90)


# Belgium had the best attacks of the tournament, the two finalists just behind

# Best attacks in Group Stage

# In[ ]:


group_attack = attacks[attacks.Round == 'Group Stage']
group_attack = group_attack.groupby('Team').sum()
group_attack = group_attack.reset_index()
group_attack = group_attack.sort_values('Goal Scored', ascending=False)


# In[ ]:


g = sns.catplot(x='Team', y='Goal Scored', data=group_attack, kind="bar", aspect=2)
g.set_xticklabels(rotation=90)


# Again, Belgium had the best attack in the Group Stage

# Mean Stats

# Goals vs Attempts

# In[ ]:


mean_matchs = data.groupby('Team').mean()
mean_matchs = mean_matchs.reset_index()


# In[ ]:


g = sns.jointplot(mean_matchs['Goal Scored'], mean_matchs['Attempts'], kind="kde", height=7)


# Best precision teams

# In[ ]:


mean_matchs['Precision'] = mean_matchs['Goal Scored'] / mean_matchs['Attempts']


# In[ ]:


g = sns.catplot(x='Team', y='Precision', data=mean_matchs.sort_values('Precision', ascending=False), kind="bar", aspect=2)
g.set_xticklabels(rotation=90)


# Russia had the best precision, in front of France and Colombia. Germany had the worst

# In[ ]:


mean_matchs.head()


# **Data Cleaning**

# Converting categorical values to label

# In[ ]:


data = pd.read_csv('../input/FIFA 2018 Statistics.csv')


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_labels = LabelEncoder()
data["Man of the Match"] = lb_labels.fit_transform(data['Man of the Match'])
data["Round"] = lb_labels.fit_transform(data['Round'])
data["PSO"] = lb_labels.fit_transform(data['PSO'])


# Removing columns with NaN values

# In[ ]:


data = data.drop(['Date', '1st Goal', 'Own goals', 'Own goal Time'], axis=1)


# In[ ]:


data = data.rename(columns = {'Man of the Match':'MoM'})


# In[ ]:


data.columns = data.columns.str.replace(" ", "_")


# In[ ]:


data.head()


# Removing the Date

# In[ ]:


data.corr()['MoM'].sort_values(ascending=False)


# Good correlations with Goal scored, the more teams score, the more they will win and shots on target

# Create X and y for the model

# In[ ]:


X = data.drop(['MoM'], axis=1)
y = data['MoM']


# Removing columns with no correlations with Man of the Match column

# In[ ]:


mom_corr = data.corr()['MoM']


# In[ ]:


mom_corr = pd.DataFrame({'col':mom_corr.index, 'correlation':mom_corr.values})


# In[ ]:


no_corr_cols = mom_corr[(mom_corr.correlation < 0.1) & (mom_corr.correlation > -0.1)]
no_corr_cols = list(no_corr_cols.col)


# In[ ]:


# Droping columns with no correlation
X = X.drop(no_corr_cols, axis=1)


# **Creating the model**

# In[ ]:


from sklearn.model_selection import train_test_split

indices = data.index.values.tolist()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.1, random_state=42)
# test_size = 0.1 because not enough to have a good precision in the model


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


# Removing teams and getting test teams
X_train = X_train.drop(['Team', 'Opponent'], axis=1)
test_teams = X_test[['Team', 'Opponent']]
X_test = X_test.drop(['Team', 'Opponent'], axis=1)


# In[ ]:


results_mom = pd.DataFrame({
    "Team": test_teams["Team"],
    "Opponent": test_teams['Opponent'],
    "MoM_true": y_test
    })


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

neighbors = range(2,10)
precision_knn = dict()

for i in neighbors:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results_mom['MoM_pred'] = y_pred
    precision = 100*round(clf.score(X_test, y_test),4)
    precision_knn[i] = precision
    print('Neighbors : ', i, '-> Precision : %s' %precision)
    
best_neighbors = max(precision_knn, key=precision_knn.get)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

n_estimators = range(2,15)
precision_rdf = dict()

for i in n_estimators:
    clf = RandomForestClassifier(n_estimators=i,random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results_mom['MoM_pred'] = y_pred
    precision = 100*round(clf.score(X_test, y_test),4)
    precision_rdf[i] = precision
    print('Estimators : ', i, '-> Precision : %s' %precision)

best_estimators = max(precision_rdf, key=precision_rdf.get)


# In[ ]:


from sklearn.linear_model import LogisticRegression

classifiers = [KNeighborsClassifier(n_neighbors=best_neighbors), LogisticRegression(random_state=42), 
               RandomForestClassifier(n_estimators=best_estimators, random_state=42)]
names = ['KNN', 'Logistic', 'Random Forest']

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results_mom['MoM_pred'] = y_pred
    precision = 100*round(clf.score(X_test, y_test),4)
    print('Model : ', name, '-> Precision : %s' %precision)


# In[ ]:


clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
results_mom['MoM_pred'] = y_pred
precision = 100*round(clf.score(X_test, y_test),4)
print('Precision : %s' %precision)


# In[ ]:


results_mom

