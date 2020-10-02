# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sklearn
from sklearn import model_selection

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.utils import resample

from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassBalance
from yellowbrick.classifier import ClassPredictionError


import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In this Kernel we were able to imporve the Recall value to 97% (but in the expense of precision)
# by downsampling the {good} transactions i.e. Class = 0. I believe that the main purpose of the model 
#(we use Random Forest) is to accurately forcasting the frauds i.e. Class = 1 even at the expense of 
#Precision value which dropped to 61%.

url = (os.path.join(dirname, filename))

df = pd.read_csv(url)
orig_df = df

# we will drop the column time as it's irrelevant in our case

df = df.drop(columns=["Time"])

# Resampling data using sklearn

mask = df.Class == 1
fraud_df = df[mask]
good_df = df[~mask]

# The next part is a for loop to find the best ratio with which we will downsample the {good} transactions vs.
# the {frauds}. Unfortunatelly, the loop takes a lot of time to run given the size of the data set. I ran it
# on my PC and found that the best ratio is 6 i.e. downsampling the {good} transactions by 6 fold relative to 
# {frauds}. We won't run it on this kernel to save time.

#ratio = int(good_df.shape[0]/fraud_df.shape[0])
#idx = []
#stat = []

#for i in range(10, ratio*10+1, 10):
#    df_downsample = resample(good_df,replace = False,n_samples = int(round(len(fraud_df)*i/10)), random_state=42)
#    df2 = pd.concat([fraud_df,df_downsample])
    #df2.math_score.value_counts()
#    X = df2.drop(columns=["Class"])
#    y = df2.Class
#    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.3, random_state=42)
#    rf = RandomForestClassifier(random_state=42)
#    rf.fit(X_train,y_train)
#    stat.append(precision_score(y_test, rf.predict(X_test))+recall_score(y_test, rf.predict(X_test)))
#    idx.append(i/10)
#    print(i)

#Optimum = pd.DataFrame(idx)
#Optimum.rename(columns={0: 'ratio'}, inplace=True)
#Optimum['stat'] = stat
# Maximum value
#opt_ratio = Optimum.iat[Optimum['stat'].argmax(),0]
# from the data the opt_ratio = 6, we will reconstruct a sample in which the data are weighted by the opt_ratio

opt_ratio = 6
df_downsample = resample(good_df,replace = False,n_samples = int(round(len(fraud_df)*opt_ratio)), random_state=42)
df2 = pd.concat([fraud_df,df_downsample])
df2.Class.value_counts()
y = df2.Class
X = df2.drop(columns=["Class"])

# splitting the new rebalanced set into train and test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.30, random_state=42)

# Random Forest (We chose Random Forest Classifier as it gave the best results)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)


# trying the same model (with the same parameters) with the bigger set of data in order to check its accuracy
y = df.Class
X = df.drop(columns=["Class"])


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.30, random_state=42)

print("Random Forest :")
print("accuracy score :", round(accuracy_score(y_test, rf.predict(X_test))*100,2),"%")
print("precision score :", round(precision_score(y_test, rf.predict(X_test))*100,2),"%")
print("recall score :", round(recall_score(y_test, rf.predict(X_test)),2)*100,"%")
print("roc_auc score :", round(roc_auc_score(y_test, rf.predict(X_test)),2)*100,"%")

# Showing confusion Matrix using yellowbrick

mapping = {0: "good", 1: "fraud"}
fig, ax = plt.subplots(figsize=(6,6))
cm_viz = ConfusionMatrix(rf, classes=["good","fraud"],label_encoder=mapping)
cm_viz.score(X_test,y_test)
cm_viz.show()

# Showing Classification Report using yellowbrick

fig, ax = plt.subplots(figsize=(6,3))
cr_viz = ClassificationReport(rf, classes=["good","fraud"],label_encoder=mapping)
cr_viz.score(X_test,y_test)
cr_viz.show()

# Showing ROC curve using yellowbrick

fig, ax = plt.subplots(figsize=(6,6))
roc_viz = ROCAUC(rf)
roc_viz.score(X_test,y_test)
roc_viz.show()

# Showing Class Balance using yellowbrick

fig, ax = plt.subplots(figsize=(6,6))
cb_viz = ClassBalance(labels=["good","fraud"])
cb_viz.fit(y_test)
cb_viz.show()

# Showing Class Prediction errors using yellowbrick

fig, ax = plt.subplots(figsize=(6,6))
cp_viz = ClassPredictionError(rf, classes=["good","fraud"])
cp_viz.score(X_test,y_test)
cp_viz.show()