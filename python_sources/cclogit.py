# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# this percentage of data is used for training the model
train_pc = 0.7

datPath = r"../input/creditcard.csv"

# load data, remove first row of names, skip first column
data = pd.read_csv(datPath,delimiter=',')

# 29 variables are used for modeling
x = data.iloc[:,1:29]
# Stardardize all the variables by remove mean and divive by standard deviation, need this to later use coeffs to measure importance of variables
x = pd.DataFrame(StandardScaler().fit_transform(x), columns = x.columns.values)

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
regr = LogisticRegression()
regr = regr.fit(xTrain, yTrain)

# Classify training and test data
trainpred = regr.predict(xTrain)
testpred = regr.predict(xTest)

# Accuracy for training and testing data
print("Train Accuracy: %f\n" % (np.mean(yTrain == trainpred) * 100))
print("Test Accuracy: %f\n" % (np.mean(yTest == testpred) * 100))

# Confusion matrix, how many instances are classified correctly (1->1, 0->0), and missclassified (0->1, 1->0)
pd.crosstab(yTest,testpred, rownames=['Actual Class'], colnames=['Predicted Class'])
#==============================================================================
# Predicted Class      0     1
# Actual Class                
# 0                83408  1887
# 1                   13   135
#==============================================================================

# Now list the coefficients of variables/features
imp = list(zip(xTrain.columns.values, abs(regr.coef_.ravel())))
#==============================================================================
#[('V1', 0.85956426521326623),
# ('V2', 0.65232589213410164),
# ('V3', 0.12454100337903135),
# ('V4', 1.5082902654525283),
# ('V5', 0.28375807258338387),
# ('V6', 0.14444370940723955),
# ('V7', 0.036545582496575753),
# ('V8', 0.83038539884427554),
# ('V9', 0.66483179276290594),
# ('V10', 1.2394576139226048),
# ('V11', 0.70693357553456826),
# ('V12', 1.3593353996434581),
# ('V13', 0.42707962825392831),
# ('V14', 1.2677904099560147),
# ('V15', 0.049473941435124291),
# ('V16', 0.55118948893255604),
# ('V17', 0.64312340696955306),
# ('V18', 0.42709361106559807),
# ('V19', 0.14806017360239038),
# ('V20', 0.092194110366660406),
# ('V21', 0.24855399821662705),
# ('V22', 0.29993733669043432),
# ('V23', 0.17872040813776172),
# ('V24', 0.079364296802087564),
# ('V25', 0.061714315121528224),
# ('V26', 0.60844677219057741),
# ('V27', 0.082083661849645412),
# ('V28', 0.066786499013850115)]
# 
#==============================================================================
# Now sort the coefficients, here use the lambda function for sorting the dictionary by value
sorted(imp, key=lambda x: x[1])
#==============================================================================
#[('V7', 0.036545582496575753),
# ('V15', 0.049473941435124291),
# ('V25', 0.061714315121528224),
# ('V28', 0.066786499013850115),
# ('V24', 0.079364296802087564),
# ('V27', 0.082083661849645412),
# ('V20', 0.092194110366660406),
# ('V3', 0.12454100337903135),
# ('V6', 0.14444370940723955),
# ('V19', 0.14806017360239038),
# ('V23', 0.17872040813776172),
# ('V21', 0.24855399821662705),
# ('V5', 0.28375807258338387),
# ('V22', 0.29993733669043432),
# ('V13', 0.42707962825392831),
# ('V18', 0.42709361106559807),
# ('V16', 0.55118948893255604),
# ('V26', 0.60844677219057741),
# ('V17', 0.64312340696955306),
# ('V2', 0.65232589213410164),
# ('V9', 0.66483179276290594),
# ('V11', 0.70693357553456826),
# ('V8', 0.83038539884427554),
# ('V1', 0.85956426521326623),
# ('V10', 1.2394576139226048),
# ('V14', 1.2677904099560147),
# ('V12', 1.3593353996434581),
# ('V4', 1.5082902654525283)]

#==============================================================================
# Now plot the decision boundary and training points using 2 most important variables
n_classes = 2
plot_colors = "br" # blue and red
plot_step = 0.02

selX1 = xTrain['V4']
selX2 = xTrain['V14']

# Train using 2 variables for plotting
regr = LogisticRegression()
regr = regr.fit(xTrain[['V4','V14']], yTrain)

x_min, x_max = selX1.min() - 1, selX1.max() + 1
y_min, y_max = selX2.min() - 1, selX2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

# ravel: Return a contiguous flattened array.
Z = regr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.xlabel('V4')
plt.ylabel('V14')
plt.axis("tight")

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(yTrain == i)
    plt.scatter(selX1.iloc[idx], selX2.iloc[idx], c=color, label=yTrain.iloc[idx].as_matrix()[:1],cmap=plt.cm.Paired)

plt.title("Decision surface of a logistic regression using paired features")
plt.legend()
plt.show()