
# data analysis and wrangling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import os
print(os.listdir("../input"))

test = os.listdir("../input")
print(test)

train_df = pd.read_csv("../input/train.csv")
# test_df = pd.read_csv('input/test.csv')
# combine = [train_df, test_df]
print("hello")
train_df.head(4)

# Any results you write to the current directory are saved as output.