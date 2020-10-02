# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get titanic & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train.info()

train['Survived'].value_counts(normalize=True)


sns.countplot(train['Survived'])
sns.plt.show()