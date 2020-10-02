# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

###### PREPARING DATA ######

# Get data frame
data = pd.read_csv("../input/diabetes.csv")
data.head()

# Drop any nulls in rows (axis 0)
data.dropna(how = 'any',axis = 0)

# Inspect data
x = data.values[:,0:8]
y = data.values[:,8]

# Check correlation
def drawCorrelation(x_points, y_points):
    plt.scatter(x_points, y_points)
    axes = plt.gca()
    m, b = np.polyfit(x_points, y_points, 1)
    x_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(x_plot, m * x_plot + b)

# Draw multiple correlations
drawCorrelation(x[:,0], x[:,1])
drawCorrelation(x[:,1], x[:,2])
drawCorrelation(x[:,3], x[:,1])

# See attributes
data.columns.values

# Drop skin column (axis 1)
data.drop(['SkinThickness'], axis = 1)

###### Molding Data ######

# Check True/False ratio
true_false_ratio = y.sum() / y.shape[0]
print(true_false_ratio)

# Drop zeros
data = data.loc[(data != 0).any(axis = 1)]

# Split training and testing data
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

x_train = train.values[:,0:8]
y_train = train.values[:,8]

x_test = test.values[:,0:8]
y_test = test.values[:,8]

###### Create and Train Model ######

# Try with gini and entropy
d_tree_1 = tree.DecisionTreeClassifier(criterion = 'gini')
d_tree_1 = d_tree_1.fit(x_train, y_train)

d_tree_2 = tree.DecisionTreeClassifier(criterion = 'entropy')
d_tree_2 = d_tree_2.fit(x_train, y_train)

###### Test Model ######

# Training accuracy

# with gini
y_pred = d_tree_1.predict(x_train)

correct = 0

for i in range(0, y_train.shape[0]):
    if y_pred[i] == y_train[i]:
        correct = correct + 1

print("training accuracy with gini: " + str(correct / y_pred.shape[0]))

# with entropy
y_pred = d_tree_2.predict(x_train)

correct = 0

for i in range(0, y_train.shape[0]):
    if y_pred[i] == y_train[i]:
        correct = correct + 1

print("training accuracy with entropy: " + str(correct / y_pred.shape[0]))


# Test accuracy

# with gini
y_pred = d_tree_1.predict(x_test)

correct = 0

for i in range(0, y_test.shape[0]):
    if y_pred[i] == y_test[i]:
        correct = correct + 1

print("validation accuracy with gini: " + str(correct / y_pred.shape[0]))

# with entropy
y_pred = d_tree_2.predict(x_test)

correct = 0

for i in range(0, y_test.shape[0]):
    if y_pred[i] == y_test[i]:
        correct = correct + 1

print("validation accuracy with entropy: " + str(correct / y_pred.shape[0]))

# Confusion matrix

print(confusion_matrix(y_test, y_pred))

# Classification report

print(classification_report(y_test, y_pred, target_names = ["actual", "predicted"]))







