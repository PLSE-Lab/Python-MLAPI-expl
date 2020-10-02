# Declaration of libraries
########################################################

# data analysis and wrangling
import numpy as np
import pandas as pd
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing/Acquiring data
########################################################

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
combine = [train, test]

# Print features available in dataset
print(train.columns.values)

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe(include=['O']))

print("\n\nInfo about training data")
train.info()

print("\n\nTop of testing data:")
print(test.head())

print("\n\nSummary statistics of testing data")
print(test.describe())

print("\n\nInfo about testing data")
test.info()

# Pivoting data
########################################################

# Pivot Pclass and Survived
print("\n\nPivoting Pclass and Survived")
print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))

# Pivot Sex and Survived
print("\n\nPivoting Sex and Survived")
print(train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False))

# Pivot SibSp and Survived
print("\n\nPivoting Siblings/Spouse and Survived")
print(train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False))

# Pivot Parch and Survived
print("\n\nPivoting Parents/Children and Survived")
print(train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False))

# Plotting data
########################################################

g = sns.FacetGrid(train,col='Survived')
g.map(plt.hist, 'Age', bins=20)



#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)