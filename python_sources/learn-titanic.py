# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# % matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# get titanic & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
print(train_df.head())
print(test_df.head())
print("testing according to other script")
print(train_df.columns.values)

print(train_df.info())
print('*'*40)
print(test_df.info())
print('*'*40)
print(train_df.describe())
print('*'*40)
print(train_df.describe(include=['O']))
print(test_df.describe())

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count().sort_values(by='Survived', ascending=False))
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))



g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)