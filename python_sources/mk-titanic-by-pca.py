import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan, precision=1)

#Print you can execute arbitrary python code
data_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
data_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

"""
print("Train Shape:")
print(data_train.shape)
print("Train Columns:")
print(data_train.keys().values)
print("Top of the training data:")
print(data_train.head())
print("Summary statistics of training data")
print(data_train.describe())
"""

# Convenience definition
driving_labels, driven_label,useless_labels =\
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked'], \
    ['Survived'], \
    ['PassengerId'] #,'Name','Ticket','Cabin'

# Split of the training data
X_train, y_train = \
    data_train.loc[:, driving_labels], \
    data_train.loc[:, driven_label]

# Split of the test data
X_test, results = \
    data_test.loc[:, driving_labels], \
    data_test.loc[:, useless_labels]

# Convenience definition
print(X_train.head())

# Better so than with the get_dummies aux call
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
X_train["Sex"] = lb.fit_transform(X_train["Sex"])
X_test["Sex"] = lb.transform(X_test["Sex"])

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

age_map = lambda a : a if not math.isnan(a) else 0.0
X_train["Age"] = X_train["Age"].map(age_map) # Needed here
X_test["Age"] = X_test["Age"].map(age_map) # Needed here?
X_train["Fare"] = X_train["Fare"].map(age_map)# NOT Needed here
X_test["Fare"] = X_test["Fare"].map(age_map) # Needed here

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_wrdn, eigen_vktrn = np.linalg.eigh(cov_mat)
print("Eigenwaarden: {}".format(eigen_wrdn))

tot = sum(eigen_wrdn)
norm_waarden = [(i/tot) for i in sorted(eigen_wrdn, reverse=True)]
cum_sum_ewdn = np.cumsum(norm_waarden)

plt.bar(range(1, 10), norm_waarden, alpha=0.5, align='center', label="Indiv. explainde variance")
plt.step(range(1, 10), cum_sum_ewdn, where='mid', label="Cumulative explained variance")
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal components")
plt.legend(loc="best")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
lr = LogisticRegression(C=1.0, random_state=1)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca, y_train)
y_test = lr.predict(X_test_pca)

results["Survived"] = y_test

results.to_csv('results_pca.csv', index=False)

print(results.head())
