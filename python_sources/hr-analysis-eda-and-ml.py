# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/HR_comma_sep.csv')

print(df.head())
print(df.keys())

print(df.groupby('left').describe())

# Quick check via Pairplot
sns.pairplot(df, hue='left')
savefig('pairplot.png')

# Jointplot showing Satisfaction versus last eval
sns.jointplot(x=df['last_evaluation'], y=df['satisfaction_level'], kind='kde')
savefig('jointplot.png')

# Satisfaction versus monthly hours
sns.lmplot(data=df, x='satisfaction_level', y='average_montly_hours', size=12, hue='left')

# Correlation
print(df.corr())

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False)
savefig('heatmap.png')

# Boxplot
g = sns.FacetGrid(df, col = 'left', size=5)
g.map(sns.boxplot, 'time_spend_company')
savefig('bp_timespend.png')

#time spend with promotion
plt.figure(figsize=(14,8))
sns.barplot(x='time_spend_company', y = 'left', hue = 'promotion_last_5years', data = df)
savefig('barplot.png')

#-----------------------------------
# MACHINE LEARNING

# data transformation
sal_dummy = pd.get_dummies(df['salary'])
df_new = pd.concat([df, sal_dummy], axis=1)
df_new.drop('salary', axis=1, inplace=True)

# Setup X and y for train_test_split
X = df_new.drop(['sales', 'left', 'high'], axis=1)
y = df_new['left']

# train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

# Import metrics
from sklearn.metrics import confusion_matrix, classification_report
print('#'*20)
print('RANDOM FOREST')
print('#'*20)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# Random Forest Score
rfc_score_train = rfc.score(X_train, y_train)
print('RFC Train Score:',rfc_score_train)
rfc_score_test = rfc.score(X_test, y_test)
print('RFC Test Score:',rfc_score_test)

# K NEAREST NEIGHBOR
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print('#'*20)
print('K NEAREST NEIGHBOR')
print('#'*20)
print(confusion_matrix(y_test, knn_pred))
print('\n')
print(classification_report(y_test, knn_pred))

# KNN Score
knn_score_train = knn.score(X_train, y_train)
print('KNN Train Score:', knn_score_train)
knn_score_test = knn.score(X_test, y_test)
print('KNN Test Score:', knn_score_test)

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(X_train, y_train)
reg_pred = lreg.predict(X_test)

print('#'*20)
print('LOGISTIC REGRESSION')
print('#'*20)
print(confusion_matrix(y_test, reg_pred))
print('\n')
print(classification_report(y_test, reg_pred))

# LOG Score

lreg_score_train = lreg.score(X_train, y_train)
print("Logistic Regression Train Score:", lreg_score_train)
lreg_score_test = lreg.score(X_test, y_test)
print('Logistic Regression Test Score:', lreg_score_test)