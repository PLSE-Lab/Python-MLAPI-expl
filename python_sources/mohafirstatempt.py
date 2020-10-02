import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

from collections import Counter

#machine learning algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

"""
#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())
"""

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

titanic_df = train.drop(['PassengerId','Name','Ticket'], axis=1)
test_df = test.drop(['Name','Ticket'], axis=1)

#Embarked
#There are only 2 missing values in train, so we replace it with most occured values.
Counter(titanic_df.Embarked)