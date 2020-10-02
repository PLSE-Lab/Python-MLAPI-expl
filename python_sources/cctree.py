# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn import tree
import sys
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_pc = 0.5
datPath = r"../input/creditcard.csv"

data = pd.read_csv(datPath,delimiter=',')

# load data, remove first row of names, skip first column
data = pd.read_csv(datPath,delimiter=',')

# 29 variables are used for modeling
x = data.iloc[:,1:29]

# the last is the target/class variable
y = data.iloc[:,30] 

m = x.shape[0]
n = x.shape[1]

# Stratified sampling: take train_pc percent from both classes (simple sampling would give worse result)
size1 = math.floor(train_pc * sum(y == 0))
size2 = math.floor(train_pc * sum(y == 1))
indY1 = np.where(y == 0)[0]
indY2 = np.where(y == 1)[0]

# Indices of instances that are used for training
# Note that the dataset is much imbalanced, so we add more/sample with replacement from the same data to make the number of instances in both classes equal
train_ind = np.concatenate((np.random.choice(indY1, size1, False),
                          np.random.choice(np.random.choice(indY2, size2, False), size1, True)), axis=0)
yTrain = y[train_ind]
test_ind = np.setdiff1d(range(m),train_ind)
yTest = y[test_ind]

# Create training and test datasets
xTrain = x.loc[train_ind, :]
xTest = x.loc[test_ind, :]

# Create and train a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xTrain, yTrain)

# Classify training and test data
trainpred = clf.predict(xTrain)
testpred = clf.predict(xTest)

# Accuracy for training and testing data
print("Train Accuracy: %f\n" % (np.mean(yTrain == trainpred) * 100))
print("Test Accuracy: %f\n" % (np.mean(yTest == testpred) * 100))

# Confusion matrix, how many instances are classified correctly (1->1, 0->0), and missclassified (0->1, 1->0)
pd.crosstab(yTest,testpred, rownames=['Actual Class'], colnames=['Predicted Class'])
# Predicted Class      0    1
# Actual Class               
# 0                85260   35
# 1                   38  110

# Now list the importance of variables/features
imp = list(zip(xTrain.columns.values, clf.feature_importances_))
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# The higher, the more important the feature. 
# The importance of a feature is computed as the (normalized) 
# total reduction of the criterion brought by that feature. 
# It is also known as the Gini importance
#==============================================================================
# [('V1', 0.0077968853534014001),
#  ('V2', 0.0043234093705722468),
#  ('V3', 0.0057487044380585672),
#  ('V4', 0.058351272917727619),
#  ('V5', 0.0061953868845815415),
#  ('V6', 0.0045615212228111903),
#  ('V7', 0.014572794030948428),
#  ('V8', 0.015834099731560396),
#  ('V9', 0.0044423361930369994),
#  ('V10', 0.018770743911557929),
#  ('V11', 0.0064203248566792482),
#  ('V12', 0.016641436534902739),
#  ('V13', 0.00039579080782825392),
#  ('V14', 0.75556782902346886),
#  ('V15', 0.00061731723924587456),
#  ('V16', 0.0025885341909976517),
#  ('V17', 0.0006622741507083228),
#  ('V18', 0.0011119274995199026),
#  ('V19', 0.0049818725620144602),
#  ('V20', 0.01515184839678786),
#  ('V21', 0.0095861344168186105),
#  ('V22', 0.0053159060681105723),
#  ('V23', 0.0082933185553981555),
#  ('V24', 0.00049646574009101449),
#  ('V25', 0.017753107738391314),
#  ('V26', 0.0052492145101938406),
#  ('V27', 0.0083469472031105513),
#  ('V28', 0.00022259645147616682)]
#==============================================================================

# Now sort the importance, here use the lambda function for sorting the dictionary by value
sorted(imp, key=lambda x: x[1])
#==============================================================================
# [('V28', 0.00022259645147616682),
#  ('V13', 0.00039579080782825392),
#  ('V24', 0.00049646574009101449),
#  ('V15', 0.00061731723924587456),
#  ('V17', 0.0006622741507083228),
#  ('V18', 0.0011119274995199026),
#  ('V16', 0.0025885341909976517),
#  ('V2', 0.0043234093705722468),
#  ('V9', 0.0044423361930369994),
#  ('V6', 0.0045615212228111903),
#  ('V19', 0.0049818725620144602),
#  ('V26', 0.0052492145101938406),
#  ('V22', 0.0053159060681105723),
#  ('V3', 0.0057487044380585672),
#  ('V5', 0.0061953868845815415),
#  ('V11', 0.0064203248566792482),
#  ('V1', 0.0077968853534014001),
#  ('V23', 0.0082933185553981555),
#  ('V27', 0.0083469472031105513),
#  ('V21', 0.0095861344168186105),
#  ('V7', 0.014572794030948428),
#  ('V20', 0.01515184839678786),
#  ('V8', 0.015834099731560396),
#  ('V12', 0.016641436534902739),
#  ('V25', 0.017753107738391314),
#  ('V10', 0.018770743911557929),
#  ('V4', 0.058351272917727619),
#  ('V14', 0.75556782902346886)]
#==============================================================================

# Now plot the decision boundary and training points using 2 most important variables
n_classes = 2
plot_colors = "br" # blue and red
plot_step = 0.02

selX1 = xTrain['V4']
selX2 = xTrain['V14']

# Train using 2 variables for plotting
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xTrain[['V4','V14']], yTrain)

x_min, x_max = selX1.min() - 1, selX1.max() + 1
y_min, y_max = selX2.min() - 1, selX2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

# ravel: Return a contiguous flattened array.
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.xlabel('V4')
plt.ylabel('V14')
plt.axis("tight")

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(yTrain == i)
    plt.scatter(selX1.iloc[idx], selX2.iloc[idx], c=color, label=yTrain.iloc[idx].as_matrix()[:1],cmap=plt.cm.Paired)

plt.title("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()
