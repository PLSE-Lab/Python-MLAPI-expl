# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassBalance
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import DiscriminationThreshold

# Input data files are available in the read-only "../input/" directory
# We input the data in pandas dataframe df

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv(os.path.join(dirname, filename))

# In this example we will trannsfrom the Math scores into binary variable {0: "failed", 1: "succeed"} and run a logit binary regression
# The model offers quite good results predicting the success and faiure of students in the math exams. The same code could be applied on
# the writing exam and the reading exam

# replace the spaces in the names of columns with '_'
df.columns = df.columns.str.replace(' ','_')
orig_df = df

# indicates the shape of the dataframe
df.shape 

# describes only the numerical columns
df.describe()

# checks if there is any missing data
df.isnull().sum()

# transforms every categorial variable to a dummi variable and gets rid of the first variable
# for example : we don't need 2 columns gender, one of them will be enough
df = pd.get_dummies(df,drop_first=True)

# we will transform the math_score to binary 0:failed, 1:succeed
for i in range(len(df.math_score)):
    if df.math_score[i]>= 50 :df.math_score[i]=1
    else : df.math_score[i]=0
        
X = df.drop(columns=["math_score","reading_score","writing_score"])
y = df.math_score

# we will divide the dataframe into training set (70%) and test set (30%)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.30, random_state=42)

# Logistic Regression

lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
confusion_matrix(y_test,lr.predict(X_test))
accuracy_score(y_test, lr.predict(X_test))
precision_score(y_test, lr.predict(X_test))
recall_score(y_test, lr.predict(X_test))
roc_auc_score(y_test, lr.predict(X_test))
print("Logistic Regression binary :")
print("accuracy score :", round(accuracy_score(y_test, lr.predict(X_test))*100,2),"%")
print("precision score :", round(precision_score(y_test, lr.predict(X_test))*100,2),"%")
print("recall score :", round(recall_score(y_test, lr.predict(X_test)),2)*100,"%")
print("roc_auc score :", round(roc_auc_score(y_test, lr.predict(X_test)),2)*100,"%")

# Showing confusion Matrix using yellowbrick

mapping = {0: "failed", 1: "succeed"}
fig, ax = plt.subplots(figsize=(6,6))
cm_viz = ConfusionMatrix(lr, classes=["failed","succeed"],label_encoder=mapping)
cm_viz.score(X_test,y_test)
cm_viz.show()

# Showing Classification Report using yellowbrick

fig, ax = plt.subplots(figsize=(6,3))
cr_viz = ClassificationReport(lr, classes=["failed","succeed"],label_encoder=mapping)
cr_viz.score(X_test,y_test)
cr_viz.show()

# Showing ROC curve using yellowbrick

fig, ax = plt.subplots(figsize=(6,6))
roc_viz = ROCAUC(lr)
roc_viz.score(X_test,y_test)
roc_viz.show()

# Showing Class Balance using yellowbrick

fig, ax = plt.subplots(figsize=(6,6))
cb_viz = ClassBalance(labels=["failed","succeed"])
cb_viz.fit(y_test)
cb_viz.show()

# Showing Class Prediction errors using yellowbrick

fig, ax = plt.subplots(figsize=(6,6))
cp_viz = ClassPredictionError(lr, classes=["failed","succeed"])
cp_viz.score(X_test,y_test)
cp_viz.show()

# Showing Discrimination Threshold using yellowbrick

fig, ax = plt.subplots(figsize=(6,5))
dt_viz = DiscriminationThreshold(lr)
dt_viz.fit(X,y)
dt_viz.score(X_test,y_test)
dt_viz.show()


