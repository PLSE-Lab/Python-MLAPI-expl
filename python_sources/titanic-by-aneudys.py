# This Python file uses the following encoding: utf-8
#Importando Cositas
import numpy as np
import pandas as pd
import random as rnd

#Visualizando
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline

#Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Analizando La Data & Agrupandola
train_dt = pd.read_csv('../input/train.csv')
test_dt = pd.read_csv('../input/test.csv')
combine = [train_dt, test_dt]

print(train_dt.columns.values)
print("\n\n")
print(test_dt.columns.values)
#Preview Data
train_dt.head()
print('_'*40)
test_dt.head()
print('_'*40)
train_dt.tail()
print('_'*40)
test_dt.tail()
print('_'*40)
train_dt.info()
print('_'*40)
test_dt.info()
print('_'*40)
print(train_dt.describe())
print('_'*40)
print(train_dt.describe(include=['O']))
print('_'*40)
print(train_dt[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False))
print('='*50)
print(train_dt[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False))
print('='*50)
print(train_dt[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False))
print('='*50)
print(train_dt[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',ascending=False))
print('='*50)

# g = sns.FacetGrid(train_dt, col='Survived', size=3, aspect =2)
# g.map(plt.hist, 'Age', color='r')
# plt.show()


# # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
# grid = sns.FacetGrid(train_dt, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();

g = sns.FacetGrid(train_dt, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# sns.plt.show()
print(sns.plt.show())







