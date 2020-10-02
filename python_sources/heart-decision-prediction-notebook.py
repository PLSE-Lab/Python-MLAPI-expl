#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "target" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 1.

# ## Importing dependencies
# We import a few dependencies like numpy, pandas, matplotlib, seaborn.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix


# ## Loading our dataset
# We load the dataset from the dataset.csv file. We load our dataset into a Pandas DataFrame object.

# In[ ]:


df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# ## Describing our dataset
# We display the first 10 entries of the DataFrame object and display the summary of the dataset.

# In[ ]:


df.head(10)


# In[ ]:


df.describe()


# Counting null values

# In[ ]:


df.isna().sum()


# In[ ]:


features = [x for x in df.columns if x != 'target']


# ## Finding relation between the target and features
# We plot different graphs to see how the target feature vary with different features.

# In[ ]:


y = df['target']
for i in features:
  x = df[i]
  plt.xlabel(i)
  plt.ylabel("Heart disease")
  plt.scatter(x, y)
  plt.show()


# ## Using heatmaps
# Graphs can give a pretty fair picture about the relationship between the targetted data and the feature. But using a heatmap shows a more accurate picture about the correlation between different features and the target variable.

# In[ ]:


plt.figure(figsize=(15, 15))
corr_mat = df.corr().round(2)
sns.heatmap(data=corr_mat, annot=True)


# ## Selecting features having good corelation factor
# We select those features which have a good correlation factor with the target variable

# In[ ]:


selected_features = ['cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


# ## Conclusion from Graphs and Heatmaps
# We shape our X and Y variables according to the selected features and target variable.

# In[ ]:


X = df[selected_features]
Y = df['target']


# ## Splitting the dataset
# We use train_test_split to test our dataset into training and testing variables.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4)


# ## Calculating TF, TN, FP, FN
# Writing a function to manually calculate the True Positives, False Positives, True Negatives and False Negatives.

# In[ ]:


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


# ## Training a KNN Model
# We try training a KNN Model for our dataset.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
scores = []
for i in range(1, 18):
  knnModel = KNeighborsClassifier(n_neighbors=i)
  knnModel.fit(X_train, Y_train)
  score = knnModel.score(x_test, y_test)
  scores.append(score)
max(scores)


# In[ ]:


errors = [(1 - x) for x in scores]
plt.figure(figsize=(8, 8))
plt.plot(range(1, 18), errors, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value') 
plt.xlabel('K') 
plt.ylabel('Error Rate')
plt.show()


# In[ ]:


knnModel = KNeighborsClassifier(n_neighbors=12)
knnModel.fit(X_train, Y_train)
knnModel.score(x_test, y_test)


# In[ ]:


plot_roc_curve(knnModel, x_test, y_test)


# In[ ]:


plot_confusion_matrix(knnModel, x_test, y_test)


# In[ ]:


y_knn_hat = knnModel.predict(x_test)


# In[ ]:


truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_knn_hat))
print("Precision is", (truePositive / (truePositive + falsePositive)))
print("Recall is", (truePositive / (truePositive + falseNegative)))
print("Specificity is", (trueNegative / (trueNegative + falsePositive)))
print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))


# ## Training a Logistic Regression Model
# We try training a Logistic Regression Model for our dataset.

# In[ ]:


from sklearn.linear_model import LogisticRegression
lrModel = LogisticRegression(max_iter=1200)
lrModel.fit(X_train, Y_train)
lrModel.score(x_test, y_test)


# In[ ]:


plot_roc_curve(lrModel, x_test, y_test)


# In[ ]:


plot_confusion_matrix(lrModel, x_test, y_test)


# In[ ]:


y_lr_hat = lrModel.predict(x_test)


# In[ ]:


truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_lr_hat))
print("Precision is", (truePositive / (truePositive + falsePositive)))
print("Recall is", (truePositive / (truePositive + falseNegative)))
print("Specificity is", (trueNegative / (trueNegative + falsePositive)))
print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))


# We see that Logistic Regression and KNN at best offer an accuracy of 80%. Let's try standard scaling our X variable and then training a KNN Model and Logistic Regression Model.

# ## Standard Scaling
# We standard scale our X variable and try to use that to increase the efficiency of our models.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for i in selected_features:
  X[i] = scaler.fit_transform(np.asarray(X[i]).reshape(-1, 1))


# ## Splitting our dataset
# We split our dataset into training and testing variables.

# In[ ]:


scaled_x_train, scaled_x_test, scaled_y_train, scaled_y_test = train_test_split(X, Y, random_state=4, test_size=0.3)


# ## Training a Logistic Regression Model
# We try training a new Logistic Regression model using the new scaled values.

# In[ ]:


newLrModel = LogisticRegression()
newLrModel.fit(scaled_x_train, scaled_y_train)


# In[ ]:


newLrModel.score(scaled_x_test, scaled_y_test)


# In[ ]:


plot_roc_curve(newLrModel, scaled_x_test, scaled_y_test)


# In[ ]:


plot_confusion_matrix(newLrModel, scaled_x_test, scaled_y_test)


# In[ ]:


y_new_lr_hat = newLrModel.predict(scaled_x_test)


# In[ ]:


truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(scaled_y_test), np.asarray(y_new_lr_hat))
print("Precision is", (truePositive / (truePositive + falsePositive)))
print("Recall is", (truePositive / (truePositive + falseNegative)))
print("Specificity is", (trueNegative / (trueNegative + falsePositive)))
print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))


# Our Logistic Regression model fares with a score of 0.78

# ## Training a KNN Model
# We try training a new KNN Model using the newly scaled values.

# In[ ]:


scores = []
for i in range(1, 8):
  newKnnModel = KNeighborsClassifier(n_neighbors=i)
  newKnnModel.fit(scaled_x_train, scaled_y_train)
  score = newKnnModel.score(scaled_x_test, scaled_y_test)
  scores.append(score)

max(scores)


# In[ ]:


errors = [(1 - x) for x in scores]
plt.figure(figsize=(8, 8))
plt.plot(range(1, 8), errors, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value') 
plt.xlabel('K') 
plt.ylabel('Error Rate')
plt.show()


# In[ ]:


newKnnModel = KNeighborsClassifier(n_neighbors=6)
newKnnModel.fit(scaled_x_train, scaled_y_train)


# Making predictions on scaled test set

# In[ ]:


y_new_knn_hat = newKnnModel.predict(scaled_x_test)


# In[ ]:


newKnnModel.score(scaled_x_test, scaled_y_test)


# Plotting a roc curve and confusion matrix for the same

# In[ ]:


plot_roc_curve(newKnnModel, scaled_x_test, scaled_y_test)


# In[ ]:


plot_confusion_matrix(newKnnModel, scaled_x_test, scaled_y_test)


# In[ ]:


truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(scaled_y_test), np.asarray(y_new_knn_hat))
print("Precision is", (truePositive / (truePositive + falsePositive)))
print("Recall is", (truePositive / (truePositive + falseNegative)))
print("Specificity is", (trueNegative / (trueNegative + falsePositive)))
print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))


# ## Visualising the model's performance
# We plot the actual data and predicted data for different selected features.

# In[ ]:


for i in selected_features:
  plt.scatter(scaled_x_test[i], scaled_y_test, color='grey')
  plt.scatter(scaled_x_test[i], y_new_knn_hat, color='red')
  plt.xlabel(i)
  plt.ylabel("Predictions")
  plt.show()


# # Conclusion
# Trained two different models, one using K Nearest Neighbors and Logistic Regression. Displayed the correlation between different features in the dataset using heatmaps and graphs. Also calculated the accuracy, specificity indicating the accuracy for both models. Also, visualised our predictions in the form of a confusion matrix and a ROC curve.
# 
# We were able to train a model with about 87% accuracy for our given dataset, thus making fair predictions for people who are more prone to heart diseases.
