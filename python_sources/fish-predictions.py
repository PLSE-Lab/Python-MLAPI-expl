#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data manipulation
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import xgboost

# Models and metrics. Since the data is small, we will test many models
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, KFold


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Import Data and Split to Train and Test

# In[ ]:


df = pd.read_csv('../input/fish-market/Fish.csv')
df.head()


# In[ ]:


df.describe(include='all')
df['Species'].value_counts()


# In[ ]:


# Split the data into features and targets
X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, shuffle=True)


# # But First, Visualize Data

# In[ ]:


plt.figure(figsize=(10,5))
plt.hist(y, bins = [0,1,2,3,4,5,6,7], align='left', rwidth=0.8, color = 'green')
plt.title('Distribution of Fish Species')
plt.xlabel('Fish Species')
plt.ylabel('Amount of Fish with Given Species')
plt.grid(b=True, axis='y')


# With a small dataset, it may be difficult to make accurate predictions. In particular, Whitefish may be difficult with under 10 entries.

# # Begin to Select Best Model
# 
# [Source [1] Code Modified from here](https://medium.com/datadriveninvestor/choosing-the-best-algorithm-for-your-classification-model-7c632c78f38f)

# In[ ]:


# Here we are creating all of our models that we want to test
# This code is modified from [1]

classifiers = []
model1 = xgboost.XGBClassifier()
classifiers.append(model1)
model2 = svm.SVC()
classifiers.append(model2)
model3 = tree.DecisionTreeClassifier(class_weight = 'balanced')
classifiers.append(model3)
model4 = RandomForestClassifier(class_weight = 'balanced')
classifiers.append(model4)
model5 = LogisticRegression(class_weight = 'balanced')
classifiers.append(model5)
model6 = KNeighborsClassifier( n_neighbors=1)
classifiers.append(model6)
model7 = KNeighborsClassifier( n_neighbors=2)
classifiers.append(model7)
model8 = KNeighborsClassifier( n_neighbors=3)
classifiers.append(model8)
model9 = KNeighborsClassifier( n_neighbors=4)
classifiers.append(model9)
model10 = KNeighborsClassifier( n_neighbors=5)
classifiers.append(model10)
model11 = GaussianNB()
classifiers.append(model11)


# In[ ]:


# This code is modified from [1]
maxAccuracy = 0
maxCV = 0
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print("Accuracy of %s is %s"%(clf, acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of %s is %s"%(clf, cm))
    
    # Get CV Score and max single score. Check if it is max for all models
    cvScore = cross_val_score(clf, X, y, cv = 6)
    cvMean = np.mean(cvScore)
    maxScore = np.max(cvScore)
    if maxAccuracy < acc:
        maxAccuracy = acc
        model = clf
    if maxCV < cvMean:
        maxCV = cvMean
        modelCV = clf
    print(cvScore)


# In[ ]:


print("Our best model was", model, "with a balanced accuracy of", maxAccuracy, ".")
print("Our best cross-validation model was", modelCV, "with the score,", maxCV, ".")


# # Conclusion and Results
# Our highest accuracy model was Logistic Regression. After tuning parameters, setting class_weights to balanced improved accuracy as it helped address our small and uneven sample distribution as seen in our graph above. We used balanced accuracy due to this inbalance in data. 
# 
# An accuracy of 90.7% is quite high and given the dataset size, is acceptable. With more data, we can expect the model to be a great predictor of fish species.
