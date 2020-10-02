import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score

sns.set(style="darkgrid")
sns.set_palette("PuBuGn_d")


# import dataset and view first five rows
data = pd.read_csv('creditcard.csv')
data.head()

# check if any feature has null values
data.isnull().values.any()

# check the imbalanced of the dataset
data['Class'].value_counts(normalize=True).rename(index = {0:'Normal:', 1:'Fraud:'})

# visualize class distribution
plt.figure(figsize=(10, 7))
data.Class.value_counts().plot(kind='bar', rot=0)
plt.title("Class distribution")
plt.xticks(range(2), ["Normal", "Fraud"])
plt.xlabel("Class")
# the dataset is extremely imbalanced, the minority class counts for around 0.002% of the examples

# define the features we want to use
X = data.ix[:,1:29]
y = data['Class']
#  I guess time & amount features will not be relevant in predicting whether or not a transaction is fraud
# so I don't think it is a useful feature that can be used for prediction

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# running the logistic regression model as it's the standard method for a binary classifier with multiple features
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, cmap='YlGnBu', annot=True, fmt="d")
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
plt.title("Confusion Matrix")

TN = conf_mat[0][0]
FN = conf_mat[1][0]
TP = conf_mat[1][1]

accuracy = 100*float(TP+TN)/float(np.sum(conf_mat))
print ('Accuracy: ' + str(np.round(accuracy, 2)) + '%')

recall = 100*float(TP)/float(TP+FN)
print ("Recall: " + str(np.round(recall, 2)) + '%')

cohen = cohen_kappa_score(y_test, y_pred)
print ('Cohen Kappa: ' + str(np.round(cohen, 3)))
# however accuracy is hight, if you look closely at the confusion matrix you will find that the model
# misclassified more than 35% of fraudulent transactions!
# so, accuracy is not the reliable measure of a model's effectiveness


# random forest is another model which perfectly suits imbalanced dataset
rf = RandomForestClassifier(n_estimators = 75, n_jobs = -1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
conf_mat = confusion_matrix(y_pred, y_test)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, cmap='YlGnBu',annot=True, fmt="d")
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
plt.title("Confusion Matrix")

TN = conf_mat[0][0]
FN = conf_mat[1][0]
TP = conf_mat[1][1]

acc = 100*float(TP+TN)/float(np.sum(conf_mat))
print ('Accuracy: ' + str(np.round(acc, 2))+ '%')

sens = 100*float(TP)/float(TP+FN)
print ('Recall: ' + str(np.round(sens, 2))+ '%')

cohen = cohen_kappa_score(y_test, y_pred)
print ('Cohen Kappa: ' + str(np.round(cohen, 3)))

# random forest model actual performs very well, note the other scores showed significant improvements
# this means there are a few features which control almost all of the behaviour here