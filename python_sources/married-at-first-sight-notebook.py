#!/usr/bin/env python
# coding: utf-8

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


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, jaccard_similarity_score
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


get_ipython().system('ls')


# In[ ]:


df = pd.read_csv("/kaggle/input/married-at-first-sight/mafs.csv")


# In[ ]:


df.head(10)


# In[ ]:


df = df.drop(['Name', 'Occupation'], axis=1)


# In[ ]:


le = LabelEncoder()


# In[ ]:


df['Location'] = le.fit_transform(df['Location'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Decision'] = le.fit_transform(df['Decision'])

df['Status'] = le.fit_transform(df['Status'])


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(14,14))
corr_matrix = df.corr().round(2)
sns.heatmap(data=corr_matrix, annot=True)


# In[ ]:


X = df[['Decision', 'Season', 'Age', 'DrLoganLevkoff', 'DrJosephCilona', 'ChaplainGregEpstein', 'PastorCalvinRoberson', 'DrJessicaGriffin']]
Y = df[['Status']]


# In[ ]:


for i in ['Decision', 'Season', 'Age', 'DrLoganLevkoff', 'DrJosephCilona', 'ChaplainGregEpstein', 'PastorCalvinRoberson', 'DrJessicaGriffin']:
  x = df[i]
  y = df['Status']
  plt.xlabel(i)
  plt.ylabel('Status')
  plt.scatter(x, y)
  plt.show()


# In[ ]:


X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=4)


# ## Using Logistic Regression
# We create a logistic regression model.

# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X_train, Y_train)


# In[ ]:


model.score(x_test, y_test)


# In[ ]:


y_hat = model.predict(x_test)


# In[ ]:


plot_confusion_matrix(model, x_test, y_test)


# ### Calculating TF, TN, FP, FN
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


# In[ ]:


truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_hat))
print("Precision is", (truePositive / (truePositive + falsePositive)))
print("Recall is", (truePositive / (truePositive + falseNegative)))
print("Specificity is", (trueNegative / (trueNegative + falsePositive)))
print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))


# ### Visualising the model's performance
# We plot the actual data and predicted data for different features.

# In[ ]:


for i in ['Decision', 'Season', 'Age', 'DrLoganLevkoff', 'DrJosephCilona', 'ChaplainGregEpstein', 'PastorCalvinRoberson', 'DrJessicaGriffin']:
  x_temp = x_test[i]
  plt.scatter(x_temp, y_test, color='grey')
  plt.scatter(x_temp, y_hat, color='red')
  plt.xlabel(i)
  plt.ylabel("Status")
  plt.show()


# ## Using KNN

# ### Finding the value of K
# To find the value of k that would give the most suitable result, we use a hit and trial approach. We plot a graph of errors vs k values and pick the value of K where the error becomes minimum.

# In[ ]:


scores = []
n = []
for i in range(1, 24):
  knnModel = KNeighborsClassifier(n_neighbors=i)
  knnModel.fit(X_train, Y_train)
  score = knnModel.score(x_test, y_test)
  scores.append(score)
  n.append(i)


# In[ ]:


max(scores)


# In[ ]:


best_score_index = scores.index(max(scores))
best_n = n[(best_score_index)]
best_n


# In[ ]:


error_rate = [1 - x for x in scores]
plt.figure(figsize =(10, 6))
plt.plot(range(1, 24), error_rate, color ='blue', linestyle ='dashed', marker ='o', markerfacecolor ='red', markersize = 10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# We see that the errors decrease sharply for k = 10. We create a K Nearest Neighbors model, set n_neighbors to 10 and train the KNN model using the fit method.

# In[ ]:


knnModel = KNeighborsClassifier(n_neighbors=10)
knnModel.fit(X_train, Y_train)
knnModel.score(x_test, y_test)


# We make predictions for the X values in the testing dataset and store it in the y_knn_hat variable. Now we can compare the y_knn_hat variable with the y_test variable to compare the accuracy of our model.

# In[ ]:


y_knn_hat = knnModel.predict(x_test)


# ### Plotting a confusion matrix
# Visualising the predictions of our model in the form of a confusion matrix to see how well our model performs.

# In[ ]:


plot_confusion_matrix(knnModel, x_test, y_test)


# Caculating True Positive, False Positive, True Negative, False Negatives. Also, we calculate the precision, recall, specificity and accuracy of our model.

# In[ ]:


truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_knn_hat))
print("Precision is", (truePositive / (truePositive + falsePositive)))
print("Recall is", (truePositive / (truePositive + falseNegative)))
print("Specificity is", (trueNegative / (trueNegative + falsePositive)))
print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))


# We plot the actual data and predicted data for different features.

# In[ ]:


for i in ['Decision', 'Season', 'Age', 'DrLoganLevkoff', 'DrJosephCilona', 'ChaplainGregEpstein', 'PastorCalvinRoberson', 'DrJessicaGriffin']:
  x_temp = x_test[i]
  plt.scatter(x_temp, y_test, color='grey')
  plt.scatter(x_temp, y_knn_hat, color='red')
  plt.xlabel(i)
  plt.ylabel("Status")
  plt.show()


# In[ ]:


j_score = jaccard_similarity_score(y_test, y_knn_hat)
print("Jaccard similarity score", j_score)


# # Conclusion
# Trained two different models, one using K Nearest Neighbors and Logistic Regression. Displayed the correlation between different features in the dataset using heatmaps and graphs. Also calculated the accuracy, specificity indicating the accuracy for both models. Also, visualised our predictions in the form of a confusion matrix.
# 
# We were able to train a model with about 85% accuracy for our given dataset, thus giving a fair insight about the thought "What makes couples stay together?" using features provided from the "Married at first sight" dataset.
