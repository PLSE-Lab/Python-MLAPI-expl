# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import re
# visulaization
#import seaborn as sns
#import matplotlib.pyplot as plt
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

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
combine = [train, test]
combine = pd.concat(combine)
print(train.columns.values)
print(combine.columns.values)
combine["HasSibSp"] = pd.Series(combine["SibSp"] > 0)
combine["HasParch"] = pd.Series(combine["Parch"] > 0)
combine["Mr."] = pd.Series(combine["Name"].str.contains("Mr\."))
combine["Mrs."] = pd.Series(combine["Name"].str.contains("Mrs\."))
combine["Miss."] = pd.Series(combine["Name"].str.contains("Miss\."))
print(combine.head())

train = combine[combine.Survived != np.nan]
test = combine[combine.Survived == np.nan]

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

print("\n\nSummary statistics of non-numerical data")
print(train.describe(include=['O']))


print("\n\nPivot - class")
# take the Pclass, survived columns of the train dataset: train[['Pclass', 'Survived']]
# group by class -- unlike sql this doesn't work because now we have one-to-many across the groupby index
# now run mean() across the many, resulting again in a one-to-one
# finally sort and pring
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n\nPivot - Sex")
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n\nPivot - SibSp")
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("\n\nCabins")
print(combine['Cabin'])

print("\n\nNames")
#print(train[["Name"]])
#print(train.Name.str.extract("([^\s]+)*", expand=False).tail())
w_freq = {}
s_freq = {}
for name in combine.Name:
    words = name.split()
    for word in words:
        if re.match("\w+,",word):
            surname = re.sub(",","",word) 
            if surname in s_freq:
                s_freq[surname] += 1
            else:
                s_freq[surname] = 1
        if word in w_freq:
            w_freq[word] += 1
        else:
            w_freq[word] = 1
words = pd.Series(w_freq)
surnames = pd.Series(s_freq)

words = words[words >= 10]
surnames = surnames[surnames >= 3]

print(words.sort_values(ascending = False))
print(surnames.sort_values(ascending = False))

print(words.index)

train["HasSibSp"] = pd.Series(train["SibSp"] > 0)
train["HasParch"] = pd.Series(train["Parch"] > 0)

    
#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)