# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrices

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Importing the dataset
df_data = pd.read_csv('../input/mushrooms.csv')

# Checking for missing values
df_data.isnull().sum()

X_org = df_data.iloc[:, 1:]  # independent variables
y_org = df_data.iloc[:, 0:1]  # dependent variables

# Check how each feature effects the target

def data_plot(hue, data):
    for i, cols in enumerate(data.columns):
        plt.figure(i)
        sns.set(rc={'figure.figsize': (12.8, 12.8)})
        ax = sns.countplot(x=data[cols], hue=hue, data=data)


# Plotting the data for dependent variable to check if any up-sampling or down-nsampling is needed
ax = sns.countplot(x=df_data['class'], data=df_data)
hue = df_data['class']
data_plot(hue, X_org)

# Label Encoding to the dependent variables
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
y_org = lb.fit_transform(y_org)

# Checking the unique number of values in each class
# print(y['Class'].value_counts())

# One Hot Encoding to the Independent Variables
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()
X_org = onehotencoder.fit_transform(X_org).toarray()

############################## LOGISTIC REGRESSION ##############################

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0)
logreg.fit(X_org, y_org.ravel())
# y_pred = logreg.predict(X_org)

y_prob = logreg.predict_proba(X_org)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)

# Confusion Matrix
confusion_matrix = metrices.confusion_matrix(y_org, y_pred)

# AUROC Curve
print("Score for Logistic Regression:",metrices.roc_auc_score(y_org, y_pred))

############################## DECISION TREE ##############################
from sklearn.tree import DecisionTreeClassifier
dt_tree = DecisionTreeClassifier(criterion='gini',max_depth=6,random_state=10)
dt_tree.fit(X_org,y_org.ravel())

y_prob_dtree = logreg.predict_proba(X_org)[:, 1]
y_pred_dtree = np.where(y_prob_dtree > 0.5, 1, 0)

# Confusion Matrix
confusion_matrix = metrices.confusion_matrix(y_org, y_pred_dtree)

# AUROC Curve
print("Score of Decision Tree",metrices.roc_auc_score(y_org, y_pred_dtree))



# Any results you write to the current directory are saved as output.