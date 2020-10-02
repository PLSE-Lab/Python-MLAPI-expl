import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# matplotlib inline

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
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# train_df.info()
# print('_'*40)
# test_df.info()

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins = 20)
# #Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(train.head())

# print("\n\nSummary statistics of training data")
# print(train.describe())

# #Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)